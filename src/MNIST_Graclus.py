import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn.pool import graclus
import data_toy
import matplotlib as cm
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    TDS = data_toy.create_dumbells()
    # TDS = ToyDataSet(edges=edges, X=X, node_labels=node_labels, node_pos=node_pos)
    # to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
    #             remove_self_loops=False):

    G = to_networkx(TDS.data)
    # graclus(edge_index, weight, num_nodes)
    # graclus(TDS.edge_index, num_nodes=10)

    fig = plt.figure(figsize=(20, 16))
    height = 1
    width = 1
    for i in range(height * width):
        # fig, ax = plt.subplots(i, figsize=(14, 12))
        # nx.draw(G, ax=ax, cmap=plt.get_cmap('Spectral'), node_color=node_colours, labels=node_labels, pos=node_pos, node_size=2000, linewidths=6)
        plt.subplot(height, width, i + 1)
        nx.draw(G, cmap=plt.get_cmap('Spectral'), node_color=TDS.X,
                pos=TDS.node_pos,
                node_size=2000)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('Spectral')),
                 cax=cbar_ax)  # https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'../graphs/Toy_{timestr}.png')