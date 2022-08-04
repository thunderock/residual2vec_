# @Filename:    link_prediction.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/01/22 8:17 PM

import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import torch
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from utils.model_utils import GraphConvolutionLayer, GraphAttentionLayer
from utils.utils import CONSTANTS
from utils.config import *
from utils.geometric_datasets import Pokec
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader

transform = T.Compose([
    T.ToUndirected(merge=True),
    T.ToDevice(DEVICE),
    T.RandomLinkSplit(num_val=.0005, num_test=.0001, is_undirected=True, add_negative_train_samples=False),
    ])

data = Pokec(root='/tmp').data
# train, test = transform(data)

train_loader = NeighborLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_neighbors=[NUM_NEIGHBORS] * 2, input_nodes=data.train_mask)
test_loader = NeighborLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_neighbors=[NUM_NEIGHBORS] * 2, input_nodes=data.test_mask)


class LinkPrediction(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=DROPOUT):
        super(LinkPrediction, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        # edge_index = edge_index.to(DEVICE)
        # x = x.to(DEVICE)
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def decode(self, z, edge_label_index):
        # cosine similarity
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)


model = LinkPrediction(in_channels=data.num_features, out_channels=128, hidden_channels=64,).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()


def train():
    model.train()
    total_examples = total_loss = 0
    for batch in tqdm(train_loader):

        optimizer.zero_grad()
        batch = batch.to(DEVICE)
        batch_size = batch.batch_size
        z = model.encode(batch.x, batch.edge_index)
        neg_edge_index = negative_sampling(edge_index=batch.edge_index, num_nodes=batch.num_nodes, num_neg_samples=None, method='sparse')
        edge_label_index = torch.cat([batch.edge_index, neg_edge_index], dim=-1,)
        edge_label = torch.cat([torch.ones(batch.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(DEVICE)
        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch_size
        total_examples += batch_size
    return total_loss / total_examples

@torch.no_grad()
def test(loader):
    model.eval()
    scores = []
    threshold = torch.tensor([.7]).to(DEVICE)
    for batch in tqdm(loader):
        batch = batch.to(DEVICE)
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_index).view(-1).sigmoid()
        pred = (out > threshold).float() * 1
        score = f1_score(np.ones(batch.edge_index.size(1)), pred.cpu().numpy())
        scores.append(score)
    return np.average(scores)


def load_model(run_id):
    """
        Returns a saved model.
    :param run_id: the model id to load
    :return: a hydrated model
    """
    if not os.path.exists(f"model_{run_id}"):
        raise Exception(f"Model id '{run_id}' does not exist.")
    model = LinkPrediction(in_channels=data.num_features, out_channels=128, hidden_channels=64).to(DEVICE)
    model.load_state_dict(torch.load(f"model_{run_id}"))
    model.eval()
    return model


def predictions(run_id, max=1000, threshold=0.99):
    """
        Creates predictions for the specified run.
    :param run_id: model id
    :param max: the maximum amount of predictions to output
    """
    pred_edges = []
    model = load_model(run_id)

    loader = NeighborLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_neighbors=[NUM_NEIGHBORS] * 2, input_nodes=None)
    threshold_tensor = torch.tensor([threshold]).to(DEVICE)
    for batch in tqdm(loader):
        batch = batch.to(DEVICE)
        z = model.encode(batch.x, batch.edge_index)
        # collecting negative edge tuples ensure that the decode are actual non-existing edges
        neg_edge_index = negative_sampling(edge_index=batch.edge_index, num_nodes = None, num_neg_samples=None, method='sparse')
        out = model.decode(z, neg_edge_index).view(-1).sigmoid()
        pred = ((out > threshold_tensor).float() * 1).cpu().numpy()
        found = np.argwhere(pred == 1)
        if found.size > 0:
            edge_tuples = neg_edge_index.t().cpu().numpy()
            select_index = found.reshape(1, found.size)[0]
            edges = edge_tuples[select_index]
            pred_edges += edges.tolist()
            if len(pred_edges) >= max:
                break

    with open(f"predictions_{run_id}.csv", "wt") as f:
        w = csv.writer(f)
        w.writerow(["source", "target"])
        for s, t in pred_edges:
            w.writerow([s, t])


def run():
    """
        Run the training and makes predictions.
    """
    run_id = int(datetime.timestamp(datetime.now()))
    writer = SummaryWriter(f"runs/{run_id}")

    start_time = datetime.now()
    epochs = 5
    with trange(epochs + 1) as t:
        for epoch in t:
            try:
                t.set_description('Epoch %i/%i train' % (epoch, epochs))
                loss = train()
                t.set_description('Epoch %i/%i test' % (epoch, epochs))
                val_acc = test(test_loader)
                t.set_postfix(loss = loss, accuracy = val_acc)
                writer.add_scalar('loss', loss, epoch)
                writer.add_scalar('accuracy', val_acc, epoch)
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {val_acc:.4f}")
            except KeyboardInterrupt:
                break
    writer.close()
    torch.save(model.state_dict(), f"model_{run_id}")
    time_elapsed = datetime.now() - start_time
    print("Creating predictions")
    predictions(run_id)
    print(f"\nRun {run_id}:")
    print(f"\tEpochs: {epoch}")
    print(f"\tTime: {time_elapsed}")
    print(f"\tAccuracy: {val_acc * 100:.01f}")
    print(f"\tParameters saved to 'model_{run_id}'.")
    print(f"\tPredictions saved to 'predictions_{run_id}.csv'.")


if __name__ == "__main__":
    run()