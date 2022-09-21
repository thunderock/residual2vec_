import numpy as np
import torch
from scipy import sparse
from scipy.sparse import csgraph, coo_matrix
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch_sparse.tensor import from_scipy


class NetworkTrainTestSplitter(object):
    def __init__(self, num_nodes: int, edge_list: torch.Tensor, fraction=0.5):
        """
        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        # # dont need graph
        # self.G = G
        # # always undirected
        # self.directed = directed
        self.num_nodes = num_nodes
        self.original_edge_set = edge_list
        self.total_number_of_edges = len(self.original_edge_set[0])
        self.number_of_test_edges = int(self.total_number_of_edges * fraction)

        self.test_edges = None
        self.train_edges = None

    def save_splitted_result(self, path_train, path_test):
        torch.save(self.train_edges, path_train)
        torch.save(self.test_edges, path_test)


class NetworkTrainTestSplitterWithMST(NetworkTrainTestSplitter):
    def __init__(self, num_nodes: int, edge_list: torch.Tensor, fraction=0.5):
        """Only support undirected Network.
        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """

        super(NetworkTrainTestSplitterWithMST, self).__init__(
            num_nodes=num_nodes, edge_list=edge_list, fraction=fraction
        )

    def _get_edge_list(self, adj: SparseTensor) -> torch.Tensor:
        row, col, _ = adj.coo()
        return torch.stack([row, col], dim=0)

    def find_mst(self, edge_list):
        row, col = edge_list
        adj = SparseTensor(row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes)).to("cpu")
        adj = adj.to_symmetric()
        MST = csgraph.minimum_spanning_tree(adj.to_scipy())
        # why isn't mst symmetric
        return from_scipy(MST).to("cpu").to_symmetric(), adj

    def train_test_split(self):
        truncated_adj, adj = self.find_mst(self.original_edge_set)
        # very slow need to do this in torch
        row, col, _ = truncated_adj.coo()
        if len(row) < self.number_of_test_edges:
            raise Exception(
                "Cannot remove edges by keeping the connectedness. Decrease the `fraction` parameter"
            )

        edge_ids = torch.from_numpy(np.random.choice(
            len(row), self.number_of_test_edges, replace=False
        ))
        remaining_adj = adj.to_scipy() - truncated_adj.to_scipy()
        self.test_edges = torch.stack([row[edge_ids], col[edge_ids]], dim=0)
        row, col = row[~edge_ids], col[~edge_ids]
        remaining_adj = remaining_adj + coo_matrix((np.ones(len(row)), (row, col)), shape=(self.num_nodes, self.num_nodes))
        self.train_edges = self._get_edge_list(from_scipy(remaining_adj))
