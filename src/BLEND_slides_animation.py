import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from networkx.classes.function import create_empty_copy

# loop through data and update plot
def update(ii, pos_t, x_t, ax, NXgraph, params):
  plt.tight_layout()
  plt.xlim([-5, 5])
  plt.xlim([-5, 5])
  pos_0 = pos_t[:, :, 0].detach().numpy()
  ax.clear()
  x_i = x_t[:, ii].detach().numpy()
  pos_i = pos_t[:, :, ii].detach().numpy()
  pos_i_dict = {i: pos_i[i, :].tolist() for i in range(pos_0.shape[0])}

  # ax.set_xlim([-3, 3])
  # ax.set_ylim([-3, 3])
  if params['kNN']:
    edge_index = knn_graph(torch.tensor(pos_i), k=params['kNN'], loop=False)
    graph = Data(edge_index=edge_index)
    NXgraph = to_networkx(graph)

  nx.draw(NXgraph, pos=pos_i_dict, ax=ax, node_size=params['node_size'],
          node_color=x_i, cmap=plt.get_cmap('Spectral'), arrows=False, width=0.25) #=params['edge_with'] )

  limits = plt.axis('on')
  ax.patch.set_edgecolor('black')
  ax.patch.set_linewidth('1')
  ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.title(f"Beltrami Flow, diffusion time={ii//10}", fontsize=16)



def create_animation(x_t, pos_t, NXgraph, params, filepath):
  # # draw initial graph
  # time = 0
  # x_0 = x_t[:, time].detach().numpy()
  # pos_0 = pos_t[:, :, time].detach().numpy()
  # pos_0_dict = {i: pos[i, :].tolist() for i in range(pos_0.shape[0])}

  fig, ax = plt.subplots(figsize=(params['im_width'], params['im_height']))
  # ax.set_xlim([-5, 5])
  # ax.set_xlim([-5, 5])
  # limits = plt.axis('on')
  # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
  # ax.set_xlim([-5, 5])
  # ax.set_xlim([-5, 5])

  # ax.axis('off')
  # nx.draw(NXgraph, pos= pos_0_dict, ax=ax, node_size=300 / 4,
  #         node_color=x, cmap=plt.get_cmap('Spectral'))
  # plt.title(f"t={time} Beltrami Flow")
  # plt.show()

  fig = plt.gcf()
  animation = FuncAnimation(fig, func=update, frames=params['T'], fargs=(pos_t, x_t, ax, NXgraph, params))
  animation.save(filepath, fps=params['fps'])  #writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=fps)


if __name__ == "__main__":
  # define graph
  # edge_index = torch.tensor([[0, 1, 1, 2],
  #                            [1, 0, 2, 1]], dtype=torch.long)
  # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
  #
  # pos = torch.tensor([[-1, -1], [0, 1], [1, 0]])

  edge_index = torch.tensor([[0, 1, 1, 2],
                             [1, 0, 2, 1]], dtype=torch.long)
  x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

  pos = torch.tensor([[-1, -1], [0, 1], [1, 0]])


  graph = Data(x=x, edge_index=edge_index, pos=pos)
  # Data(edge_index=[2, 4], x=[3, 1])

  # draw initial graph
  NXgraph = to_networkx(graph)
  # pos_dict = {i: pos[i, :].tolist() for i in range(pos.shape[0])}
  # nx.draw(NXgraph, pos=pos_dict, node_color=x, cmap=plt.get_cmap('Spectral'))
  # plt.show()

  # nx.draw(NXgraph, r_centroids, ax=ax, node_size=300 / 4, edge_color=list(edge_weights),  # "lime",
  #         node_color=x, cmap=plt.get_cmap('Spectral'), width=list(edge_weights))

  # define evolutions
  x_t = torch.tensor([[-1, -2, -3], [0, 0, 0], [1, 2, 3]], dtype=torch.float)  # [N, T]
  pos_t = torch.tensor([[[-1, -2, -3], [-1, -1, -1]], [[0, 0, 0], [1, 2, 3]], [[1, 0, -1], [0, 1, 2]]])  # [N, 2, T]
  max_pos = pos_t.max()
  # pos_t = pos_t / (2 * max_pos) + torch.tensor([0.5])
  im_height = 16
  im_width = 16
  frames = 3
  fps = 1.5
  filepath = f"../BLEND_animation/BLEND_animation.gif"
  params= {}
  # , im_height = 16, im_width = 16, frames = 3, fps = 1.5
  create_animation(x_t, pos_t, NXgraph, params, filepath)