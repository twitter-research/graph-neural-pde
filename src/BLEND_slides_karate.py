import torch
from torch_geometric.datasets import KarateClub
import networkx as nx
from torch.distributions.multivariate_normal import MultivariateNormal

dataset = KarateClub()

for i in dataset[0]:
  print(i)
# this torch.geometric.datasets object comprises of edge(edge information for each node), x(nodes) and y(labels for each node)

data = dataset[0]

new_labels = torch.tensor([0 if i in [0,1] else 1 for i in data.y.tolist()])
pos_mean_0 = torch.tensor([-1.0, 0.0])
pos_sd_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
pos_mean_1 = torch.tensor([0.0, 1.0])
pos_sd_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
MN_0 = MultivariateNormal(pos_mean_0, pos_sd_0)
MN_1 = MultivariateNormal(pos_mean_1, pos_sd_1)
start_pos = torch.stack([MN_0.rsample(sample_shape=torch.Size([1])) if i==0 else MN_1.rsample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])
torch.distributions.multivariate_normal.MultivariateNormal(pos_mean_0, pos_sd_0)
end_pos_sd_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
end_pos_sd_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
MNp_0_end = MultivariateNormal(pos_mean_0, end_pos_sd_0)
MNp_1_end = MultivariateNormal(pos_mean_1, end_pos_sd_1)
end_pos = torch.stack([MNp_0_end.rsample(sample_shape=torch.Size([1])) if i==0 else MNp_1_end.rsample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])

feat_mean_0 = torch.tensor([-1.0])
feat_sd_0 = torch.tensor([1.0])
feat_mean_1 = torch.tensor([1.0])
feat_sd_1 = torch.tensor([1.0])
MNf_0 = MultivariateNormal(feat_mean_0, feat_sd_0)
MNf_1 = MultivariateNormal(feat_mean_1, feat_sd_1)
start_feats  = torch.stack([MN_0.rsample(sample_shape=torch.Size([1])) if i==0 else MN_1.rsample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])
#############CHECK here

end_feat_sd_0 = feat_sd_0 * 0.3
end_feat_sd_1 = feat_sd_1 * 0.3
end_feats = torch.stack([MNp_0_end.rsample(sample_shape=torch.Size([1])) if i==0 else MNp_1_end.rsample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])

N = len(data.y)
T = 20

pos_t = torch.zeros((N,2,T))
for t in range(T):
    pos_t[N, 2, t] = start_feats + (end_feats - start_feats) * t / T


data.new_labels = new_labels
data.post_t = pos_t

numpyx = data.x[1].numpy()
numpyy = data.y.numpy()
numpyedge = edge[1].numpy()

g = nx.Graph(numpyx)

name,edgeinfo = edge

src = edgeinfo[0].numpy()
dst = edgeinfo[1].numpy()
edgelist = zip(src,dst)

for i,j in edgelist:
  g.add_edge(i,j)

nx.draw_networkx(g)