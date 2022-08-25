from utils.link_prediction import *
from datasets import triplet_dataset
from utils.geometric_datasets import Pokec
from torch_geometric.loader import NeighborLoader

d = triplet_dataset.TripletPokecDataset()
ds = triplet_dataset.NeighborEdgeSampler(d, batch_size=4, shuffle=True, num_workers=1)
batch = next(iter(ds))
a, p, n = batch


# d = Pokec().data
# ds = NeighborLoader(d, batch_size=4, shuffle=True, num_neighbors=[10, 10], )
# batch = next(iter(ds))
# node = batch.x
# edge_list = batch.edge_index


model = GATLinkPrediction(in_channels=d.num_features, embedding_size=128, num_heads=2, num_layers=3, hidden_channels=64, )

x = model.forward_i(a[0], a[1])
y1 = model.forward_o(p[0], p[1])
y2 = model.forward_o(n[0], n[1])
Y1 = model.decode(x, y1)
Y2 = model.decode(x, y2)

print(x.shape, y1.shape, y2.shape, Y1, Y2)

model = GCNLinkPrediction(in_channels=d.num_features, embedding_size=128, hidden_channels=64, num_layers=3, )

x = model.forward_i(a[0], a[1])
y1 = model.forward_o(p[0], p[1])
y2 = model.forward_o(n[0], n[1])
Y1 = model.decode(x, y1)
Y2 = model.decode(x, y2)

print(x.shape, y1.shape, y2.shape, Y1, Y2)



