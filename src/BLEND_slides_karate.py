import numpy as np
import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data
import networkx as nx
from torch.distributions.multivariate_normal import MultivariateNormal
from BLEND_slides_animation import create_animation
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt

dataset = KarateClub()
data = dataset[0]

N = len(data.y)
params = {'T':100, 'fps':10, 'end_feat_sd_scale':0.1, 'end_pos_sd_scale':0.05,
          'node_size': 500, 'edge_width':0.25, 'im_height':8, 'im_width':16, 'kNN':True}

T = params['T']

new_labels = torch.tensor([0 if i in [0,1] else 1 for i in data.y.tolist()])
# position
pos_mean_0 = torch.tensor([-1.0, 0.0])
pos_sd_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
pos_mean_1 = torch.tensor([1.0, 1.0])
pos_sd_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
MN_0 = MultivariateNormal(pos_mean_0, pos_sd_0)
MN_1 = MultivariateNormal(pos_mean_1, pos_sd_1)
# start_pos = torch.stack([MN_0.rsample(sample_shape=torch.Size([1])) if i==0 else MN_1.rsample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])
# start_pos = torch.stack([MN_0.sample(sample_shape=torch.Size([1])) if i==0 else MN_1.sample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])
start_pos = torch.stack([torch.tensor(np.random.multivariate_normal(pos_mean_0, pos_sd_0)) if i==0 else torch.tensor(np.random.multivariate_normal(pos_mean_1, pos_sd_1)) for i in new_labels.tolist()])

end_pos_sd_0 = pos_sd_0 * params['end_pos_sd_scale'] #torch.tensor([[1.0, 0.0], [0.0, 1.0]])
end_pos_sd_1 = pos_sd_0 * params['end_pos_sd_scale'] #torch.tensor([[1.0, 0.0], [0.0, 1.0]])
MNp_0_end = MultivariateNormal(pos_mean_0, end_pos_sd_0)
MNp_1_end = MultivariateNormal(pos_mean_1, end_pos_sd_1)
# end_pos = torch.stack([MNp_0_end.rsample(sample_shape=torch.Size([1])) if i==0 else MNp_1_end.rsample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])
end_pos = torch.stack([MNp_0_end.sample(sample_shape=torch.Size([1])) if i==0 else MNp_1_end.sample(sample_shape=torch.Size([1])) for i in new_labels.tolist()])
#features
feat_mean_0 = torch.tensor([-1.0])
feat_sd_0 = torch.tensor([1.0])
feat_mean_1 = torch.tensor([1.0])
feat_sd_1 = torch.tensor([1.0])
start_feat  = torch.stack([torch.normal(feat_mean_0, feat_sd_0) if i==0 else torch.normal(feat_mean_1, feat_sd_1) for i in new_labels.tolist()])
end_feat_sd_0 = feat_sd_0 * params['end_feat_sd_scale']
end_feat_sd_1 = feat_sd_1 * params['end_feat_sd_scale']
end_feat  = torch.stack([torch.normal(feat_mean_0, end_feat_sd_0) if i==0 else torch.normal(feat_mean_1, end_feat_sd_1) for i in new_labels.tolist()])

# build time series with linear interp
pos_t = torch.zeros((N,2,T)) #[34,2,20]
feat_t = torch.zeros((N,T)) #[34,20]
# test_start_pos_t = torch.zeros((N,2,T)) #[34,2,20]
# test_end_pos_t = torch.zeros((N,2,T)) #[34,2,20]

for t in range(T):
    # print(end_pos.squeeze().T)
    # print(start_pos.squeeze()[0,:])
    # print(end_pos.squeeze()[0,:])
    # pos_t[:, :, t] = start_pos.squeeze() + (end_pos.squeeze() - start_pos.squeeze()) * t / T
    # feat_t[:, t] = start_feat.squeeze() + (end_feat.squeeze() - start_feat.squeeze()) * t / T
    f = T / np.log(T)
    s = f * np.log(1+min(t,0.8*T))
    pos_t[:, :, t] = start_pos.squeeze() + (end_pos.squeeze() - start_pos.squeeze()) * s / T
    feat_t[:, t] = start_feat.squeeze() + (end_feat.squeeze() - start_feat.squeeze()) * s / T

    # test_start_pos_t[:, :, t] = start_pos.squeeze()
    # test_end_pos_t[:, :, t] = end_pos.squeeze()

data.new_labels = new_labels
data.post_t = pos_t
data.feat_t = feat_t
data.x = start_feat.squeeze()

filepath = f"../BLEND_animation/karate_animation.gif"

graph = Data(x=start_feat.squeeze(), edge_index=data.edge_index)
NXgraph = to_networkx(graph)
create_animation(feat_t, pos_t, NXgraph, params, filepath)

#TESTS
# fig, ax = plt.subplots()
# plt.plot(pos_t[:,0,:].T)
# plt.title("pos_t x")
# plt.show()
#
# fig, ax = plt.subplots()
# plt.plot(test_start_pos_t[:,0,:].T)
# plt.title("test_start_pos_t x")
# plt.show()
#
# fig, ax = plt.subplots()
# plt.plot(test_end_pos_t[:,0,:].T)
# plt.title("test_end_pos_t x")
# plt.show()
#
# fig, ax = plt.subplots()
# plt.plot(test_start_pos_t[:,1,:].T)
# plt.title("test_start_pos_t y")
# plt.show()
#
# fig, ax = plt.subplots()
# plt.plot(test_end_pos_t[:,1,:].T)
# plt.title("test_end_pos_t y")
# plt.show()
