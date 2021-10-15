import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data
import networkx as nx
from torch.distributions.multivariate_normal import MultivariateNormal
from BLEND_slides_animation import create_animation
from torch_geometric.utils.convert import to_networkx

dataset = KarateClub()
data = dataset[0]

new_labels = torch.tensor([0 if i in [0,1] else 1 for i in data.y.tolist()])
# position
pos_mean_0 = torch.tensor([-1.0, 0.0])
pos_sd_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
pos_mean_1 = torch.tensor([0.0, 1.0])
pos_sd_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
MN_0 = MultivariateNormal(pos_mean_0, pos_sd_0)
MN_1 = MultivariateNormal(pos_mean_1, pos_sd_1)
start_pos = torch.stack([MN_0.rsample(sample_shape=torch.Size([1])) if i==0 else MN_1.rsample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])
end_pos_sd_0 = pos_sd_0 * 0.2 #torch.tensor([[1.0, 0.0], [0.0, 1.0]])
end_pos_sd_1 = pos_sd_0 * 0.2 #torch.tensor([[1.0, 0.0], [0.0, 1.0]])
MNp_0_end = MultivariateNormal(pos_mean_0, end_pos_sd_0)
MNp_1_end = MultivariateNormal(pos_mean_1, end_pos_sd_1)
end_pos = torch.stack([MNp_0_end.rsample(sample_shape=torch.Size([1])) if i==0 else MNp_1_end.rsample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])
#features
feat_mean_0 = torch.tensor([-1.0])
feat_sd_0 = torch.tensor([1.0])
feat_mean_1 = torch.tensor([1.0])
feat_sd_1 = torch.tensor([1.0])
start_feat  = torch.stack([torch.normal(feat_mean_0, feat_sd_0) if i==0 else torch.normal(feat_mean_1, feat_sd_1) for i in new_labels.tolist()])
end_feat_sd_0 = feat_sd_0 * 0.1
end_feat_sd_1 = feat_sd_1 * 0.1
end_feat  = torch.stack([torch.normal(feat_mean_0, end_feat_sd_0) if i==0 else torch.normal(feat_mean_1, end_feat_sd_1) for i in new_labels.tolist()])

N = len(data.y)
T = 20
# build time series with linear interp
pos_t = torch.zeros((N,2,T)) #[34,2,20]
feat_t = torch.zeros((N,T)) #[34,20]
for t in range(T):
    pos_t[:, :, t] = start_pos.squeeze() + (end_pos.squeeze() - start_pos.squeeze()) * t / T
    feat_t[:, t] = start_feat.squeeze() + (end_feat.squeeze() - start_feat.squeeze()) * t / T

data.new_labels = new_labels
data.post_t = pos_t
data.feat_t = feat_t
data.x = start_feat.squeeze()

im_height = 16
im_width = 16
frames = 20
fps = 1.5
filepath = f"../BLEND_animation/karate_animation.gif"

graph = Data(x=start_feat.squeeze(), edge_index=data.edge_index)
NXgraph = to_networkx(graph)
create_animation(feat_t, pos_t, NXgraph, filepath, im_height, im_width, frames, fps)