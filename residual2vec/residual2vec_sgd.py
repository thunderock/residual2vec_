"""A python implementation of residual2vec based on the stochastic gradient
descent algorithm. Suitable for large networks.

Usage:

```python
import residual2vec as rv

# Node sampler for the noise distribution for negative sampling
noise_sampler = rv.ConfigModelNodeSampler()

model = rv.residual2vec_sgd(noise_sampler = noise_sampler, window_length = 10)
model.fit(G)
emb = model.transform(dim = 64)
# or equivalently emb = model.fit(G).transform(dim = 64)
```

If want to remove the structural bias associated with node labels (i.e., gender):
```python
import residual2vec as rv

group_membership = [0,0,0,0,1,1,1,1] # an array of group memberships of nodes.

# SBMNodeSampler reflects the group membership in sampling
noise_sampler = SBMNodeSampler(window_length = 10, group_membership = group_membership)

model = rv.residual2vec_matrix_factorization(noise_sampler, window_length = 10)
model.fit(G)
emb = model.transform(dim = 64)
```

You can customize the noise_sampler by implementing the following class:

```python
import residual2vec as rv
class CustomNodeSampler(rv.NodeSampler):
    def fit(self, A):
        #Fit the sampler
        #:param A: adjacency matrix
        #:type A: scipy.csr_matrix
        pass

    def sampling(self, center_node, n_samples):
        #Sample context nodes from the graph for center nodes
        #:param center_node: ID of center node
        #:type center_node: int
        #:param n_samples: number of samples per center node
        #:type n_samples: int
        pass
```
"""
import random

import numpy as np
from numba import njit
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy import sparse
from residual2vec import utils
from residual2vec.random_walk_sampler import RandomWalkSampler
from residual2vec.word2vec import NegativeSampling
import wandb
from utils.config import *

class residual2vec_sgd:
    """Residual2Vec based on the stochastic gradient descent.

    .. highlight:: python
    .. code-block:: python
        >>> from residual2vec.residual2vec_sgd import residual2vec_sgd
        >>> net = nx.karate_club_graph()
        >>> model = r2v.Residual2Vec(null_model="configuration", window_length=5, restart_prob=0, residual_type="individual")
        >>> model.fit(net)
        >>> in_vec = model.transform(net, dim = 30)
        >>> out_vec = model.transform(net, dim = 30, return_out_vector=True)
    """

    def __init__(
        self,
        noise_sampler,
        window_length=10,
        batch_size=256,
        num_walks=5,
        walk_length=40,
        p=1,
        q=1,
        cuda=DEVICE,
        buffer_size=100000,
        context_window_type="double",
        miniters=200,
    ):
        self.window_length = window_length
        self.sampler = noise_sampler
        self.cuda = cuda
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.miniters = miniters
        self.context_window_type = context_window_type

    # add feature matrix here
    def fit(self, adjmat=None):
        if not (sparse.isspmatrix(adjmat) or isinstance(adjmat, np.ndarray)):
            # dont need sampler in case of link prediction
            self.n_nodes = None
            return self
        # Convert to scipy.sparse.csr_matrix format
        adjmat = utils.to_adjacency_matrix(adjmat)

        # Set up the graph object for efficient sampling
        self.adjmat = adjmat
        self.n_nodes = adjmat.shape[0]
        self.sampler.fit(adjmat)
        return self

    def transform(self, model, dataloader: torch.utils.data.DataLoader, epochs=1):
        """
        * model is the model to be used with the framework
        * x are the node features
        """

        # Set up the embedding model
        PADDING_IDX = self.n_nodes if self.n_nodes else dataloader.dataset.n_nodes
        # model = Word2Vec(
        #     vocab_size=self.n_nodes + 1, embedding_size=dim, padding_idx=PADDING_IDX
        # )
        neg_sampling = NegativeSampling(embedding=model)
        model.to(self.cuda)
        # Training
        optim = Adam(model.parameters(), lr=0.003)

        # number of batches
        n_batches = len(dataloader)
        patience_threshold = int(n_batches * .5) # 50% of the batches
        print(f"Patience threshold: {patience_threshold}")
        for epoch in range(epochs):
            break_loop = False
            patience = 0
            pbar = tqdm(dataloader, miniters=self.miniters)
            batch_num = 0
            for iword, owords, nwords in pbar:
                optim.zero_grad()
                for param in model.parameters():
                    param.grad = None
                loss = neg_sampling(iword, owords, nwords)
                if not torch.is_nonzero(loss):
                    patience += 1
                    if patience > patience_threshold:
                        break_loop = True
                        print("Early stopping {}, patience: {}".format(epoch, patience))
                        break
                loss.backward()
                optim.step()
                wandb.log({"epoch": epoch, "loss": loss.item(), "batch_num": batch_num})
                pbar.set_postfix(epoch=epoch, loss=loss.item())
                batch_num += 1
            if break_loop:
                break
        self.in_vec = model.ivectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        self.out_vec = model.ovectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        return self.in_vec


class TripletSimpleDataset(Dataset):
    """Dataset for training word2vec with negative sampling."""

    def __init__(
        self,
        adjmat,
        num_walks,
        window_length,
        noise_sampler,
        padding_id,
        walk_length=40,
        p=1.0,
        q=1.0,
        context_window_type="double",
        buffer_size=100000,
    ):

        self.adjmat = adjmat
        # self.num_features = 1
        # self.X = torch.from_numpy(group_ids).unsqueeze(-1).to(torch.float32)
        self.num_walks = num_walks
        # rows, cols = self.adjmat.nonzero()
        # self.edge_index = torch.from_numpy(np.stack([rows, cols], axis=0)).to(torch.int64)
        # self.neg_edge_index = negative_sampling(edge_index=self.edge_index, num_nodes=self.X.shape[0],
        #                                         num_neg_samples=None, method='sparse', force_undirected=True)
        self.window_length = window_length
        self.noise_sampler = noise_sampler
        self.walk_length = walk_length
        self.padding_id = padding_id
        self.context_window_type = {"double": 0, "left": -1, "right": 1}[
            context_window_type
        ]
        self.rw_sampler = RandomWalkSampler(
            adjmat, walk_length=walk_length, p=p, q=q, padding_id=padding_id
        )
        self.node_order = np.random.choice(
            adjmat.shape[0], adjmat.shape[0], replace=False
        )
        self.n_nodes = adjmat.shape[0]

        self.ave_deg = adjmat.sum() / adjmat.shape[0]

        # Counter and Memory
        self.n_sampled = 0
        self.sample_id = 0
        self.scanned_node_id = 0
        self.buffer_size = buffer_size
        self.contexts = None
        self.centers = None
        self.random_contexts = None
        # self.num_embeddings = len(np.unique(group_ids)) + 1

        # Initialize
        self._generate_samples()

    def __len__(self):
        return self.n_nodes * self.num_walks * self.walk_length

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self._generate_samples()

        center = self.centers[self.sample_id]
        cont = self.contexts[self.sample_id].astype(np.int64)
        rand_cont = self.random_contexts[self.sample_id].astype(np.int64)

        self.sample_id += 1

        return center, cont, rand_cont

    def _generate_samples(self):
        next_scanned_node_id = np.minimum(
            self.scanned_node_id + self.buffer_size, self.n_nodes
        )
        walks = self.rw_sampler.sampling(
            self.node_order[self.scanned_node_id : next_scanned_node_id]
        )
        self.centers, self.contexts = _get_center_context(
            context_window_type=self.context_window_type,
            walks=walks,
            n_walks=walks.shape[0],
            walk_len=walks.shape[1],
            window_length=self.window_length,
            padding_id=self.padding_id,
        )
        self.random_contexts = self.noise_sampler.sampling(
            center_nodes=self.centers,
            context_nodes=self.contexts,
            padding_id=self.padding_id,
        )
        self.n_sampled = len(self.centers)
        self.scanned_node_id = next_scanned_node_id % self.n_nodes
        self.sample_id = 0


def _get_center_context(
    context_window_type, walks, n_walks, walk_len, window_length, padding_id
):
    """Get center and context pairs from a sequence
    window_type = {-1,0,1} specifies the type of context window.
    window_type = 0 specifies a context window of length window_length that extends both
    left and right of a center word. window_type = -1 and 1 specifies a context window
    that extends either left or right of a center word, respectively.
    """
    if context_window_type == 0:
        center, context = _get_center_double_context_windows(
            walks, n_walks, walk_len, window_length, padding_id
        )
    elif context_window_type == -1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
        )
    elif context_window_type == 1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=False
        )
    else:
        raise ValueError("Unknown window type")
    center = np.outer(center, np.ones(context.shape[1]))
    center, context = center.reshape(-1), context.reshape(-1)
    s = (center != padding_id) * (context != padding_id)
    center, context = center[s], context[s]
    order = np.arange(len(center))
    random.shuffle(order)
    return center[order].astype(int), context[order].astype(int)


@njit(nogil=True)
def _get_center_double_context_windows(
    walks, n_walks, walk_len, window_length, padding_id
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones(
        (n_walks * walk_len, 2 * window_length), dtype=np.int64
    )
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        for i in range(window_length):
            if t_walk - 1 - i < 0:
                break
            contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]

        for i in range(window_length):
            if t_walk + 1 + i >= walk_len:
                break
            contexts[start:end, window_length + i] = walks[:, t_walk + 1 + i]

    return centers, contexts


@njit(nogil=True)
def _get_center_single_context_window(
    walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones((n_walks * walk_len, window_length), dtype=np.int64)
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        if is_left_window:
            for i in range(window_length):
                if t_walk - 1 - i < 0:
                    break
                contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]
        else:
            for i in range(window_length):
                if t_walk + 1 + i >= walk_len:
                    break
                contexts[start:end, i] = walks[:, t_walk + 1 + i]
    return centers, contexts

