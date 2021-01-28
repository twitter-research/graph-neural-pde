from torch_geometric.data import Data, InMemoryDataset, download_url, DataLoader
from data_image import edge_index_calc
import torch
import numpy as np


class ToyDataSet():
    def __init__(self, edges, X, node_labels, node_pos, num_classes=2):
        self.edges = edges
        self.X = X
        self.data = Data(x=self.X, edge_index=self.edges, num_nodes = self.X.shape[0])
        self.node_labels = node_labels
        self.node_pos = node_pos
        self.num_classes = num_classes

def create_dumbells():
    # adj = np.array([
    #     [0, 1, 1, 0, 0, 0, 0, 0],
    #     [1, 0, 0, 1, 0, 0, 0, 0],
    #     [1, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 1, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 1, 0]])
    src = [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7]
    dst = [1, 2, 0, 3, 0, 3, 1, 2, 4, 3, 5, 4, 6, 7, 5, 7, 5, 6]
    edges = torch.tensor([src, dst])
    X = torch.tensor(np.resize(np.array(range(16)), (8, 2)), dtype=torch.float)

    # weights = [8] * len(src)
    node_labels = {i: i for i in range(8)}
    node_pos_list = [[0, 0], [1, 1], [1, -1], [2, 0], [3, 0], [4, 0], [5, 1], [5, -1]]
    node_pos = {i: node_pos_list[i] for i in range(8)}

    TDS = ToyDataSet(edges=edges, X=X, node_labels=node_labels, node_pos=node_pos)
    return TDS

def create_grid(im_height, im_width, diags):
    edge_index = edge_index_calc(im_height, im_width, diags)
    X = torch.tensor(np.resize(np.array(range(im_height*im_width*2)), (im_width*im_width, 2)), dtype=torch.float)
    node_labels = {i: i for i in range(im_height*im_width)}

    node_pos = {}
    for y in range(im_height,-1,-1):
        for x in range(im_width):
            node_pos[y*im_height + x] = [x,im_height-y-1]

    TDS = ToyDataSet(edges=edge_index, X=X, node_labels=node_labels, node_pos=node_pos)
    return TDS

def create_eggtimer(im_height, im_width, diags):
    im_height = 8
    im_width = 8
    edge_index = edge_index_calc(im_height, im_width, diags)
    node_labels = {i: i for i in range(im_height*im_width)}

    rem_list = [16,17,18,21,22,23]
    remove_list = []
    for i in range(4):
        remove_list.extend([j+i*im_width for j in rem_list])

    #dict for reindexing
    convert_dict = {}
    counter = 0
    for i in range(im_height*im_width):
        if i in remove_list:
            pass
        else:
            convert_dict[i] = counter
            counter += 1

    node_pos = {}
    for y in range(im_height,-1,-1):
        for x in range(im_width):
            if y*im_height + x in remove_list:
                del node_labels[y*im_height + x] #delete excess node lables
            else:
                node_pos[y*im_height + x] = [x,im_height-y-1] #add remaining node positions

    new_node_pos = {}
    #reindex node_pos
    for node in node_pos:
        if node in convert_dict:
            new_node_pos[convert_dict[node]] = node_pos[node]
    #reindex node_labels
    new_node_labels = {}
    for node in node_labels:
        if node in convert_dict:
            new_node_labels[convert_dict[node]] = node_labels[node]

    #delete excess edges
    new_edges = []
    for edge in range(len(edge_index[0])):
        if edge_index[0][edge] in remove_list or edge_index[1][edge] in remove_list:
            pass
        else:
            # new_edges.append([edge_index[0][edge], edge_index[1][edge]])
            new_edges.append([convert_dict[edge_index[0][edge].item()], convert_dict[edge_index[1][edge].item()]])

    new_edges = torch.tensor(new_edges).T


    #x on remaning nodes only
    X = torch.tensor(np.resize(np.array(range((im_height*im_width-4*len(rem_list))*2)),
                               (im_width*im_width-4*len(rem_list), 2)), dtype=torch.float)

    TDS = ToyDataSet(edges=new_edges, X=X, node_labels=new_node_labels, node_pos=new_node_pos)
    return TDS

def create_4corners(im_height, im_width, diags):
    im_height = 4
    im_width = 4

    edge_index_list = []
    edge_index = edge_index_calc(im_height, im_width, diags=True)
    for i in range(4):
        edge_index_list.append(edge_index + i * im_height * im_width)

    squares = torch.cat(edge_index_list,dim=1)#torch.tensor()
    downdiag = torch.tensor([[15,64,65,66,67,68,64,65,66,67,68,48],[64,65,66,67,68,48,15,64,65,66,67,68]])
    updiag = torch.tensor([[35,69,70,66,71,72,69,70,66,71,72,28], [69,70,66,71,72,28,35,69,70,66,71,72]])
    edge_index = torch.cat([squares,downdiag,updiag],dim=1)

    node_pos = {}
    shift = [0,6]
    shifts = [[0,3],[9,3],[0,-6],[9,-6]]
    for i in range(4):
        for y in range(im_height):
            for x in range(im_width):
                node = i * im_height * im_width + y * im_height + x
                pos = [x + shift[0] + shifts[i][0], im_height - y - 1 + shift[1] + shifts[i][1]]
                node_pos[node] = pos
                print(str(node) + " " + str(pos))
    for i in range(64, 69):
        node = i
        pos = [shift[0] + im_width + i - 64, shift[1] - im_height - (i - 64)+6]
        node_pos[node] = pos
        print(str(node) + " " + str(pos))

    node_pos[69] = [4,4]
    node_pos[70] = [5,5]
    node_pos[71] = [7,7]
    node_pos[72] = [8,8]

    node_labels = {i: i for i in range(im_height*im_width*4+5+4)}

    X = torch.tensor([node_pos[i] for i in range(im_height*im_width*4+5+4)],dtype=torch.float)
    TDS = ToyDataSet(edges=edge_index, X=X, node_labels=node_labels, node_pos=node_pos)
    return TDS

# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from colorspacious import cspace_converter
# from collections import OrderedDict
#
# mpl.rcParams.update({'font.size': 14})
# # Indices to step through colormap.
# x = np.linspace(0.0, 1.0, 100)
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))
#
# def plot_color_gradients(cmap_category, cmap_list):
#     fig, axes = plt.subplots(nrows=len(cmap_list), ncols=2)
#     fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99,
#                         wspace=0.05)
#     fig.suptitle(cmap_category + ' colormaps', fontsize=14, y=1.0, x=0.6)
#
#     for ax, name in zip(axes, cmap_list):
#
#         # Get RGB values for colormap.
#         rgb = cm.get_cmap(plt.get_cmap(name))(x)[np.newaxis, :, :3]
#
#         # Get colormap in CAM02-UCS colorspace. We want the lightness.
#         lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
#         L = lab[0, :, 0]
#         L = np.float32(np.vstack((L, L, L)))
#
#         ax[0].imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
#         ax[1].imshow(L, aspect='auto', cmap='binary_r', vmin=0., vmax=100.)
#         pos = list(ax[0].get_position().bounds)
#         x_text = pos[0] - 0.01
#         y_text = pos[1] + pos[3]/2.
#         fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
#
#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axes.flat:
#         ax.set_axis_off()
#     # plt.show()
#     plt.savefig(f'../graphs/cmaps.png')



if __name__ == "__main__":
    # dumbells = create_dumbells()
    # print(dumbells.edges)
    cmaps = OrderedDict()
    # cmaps['Diverging'] = [
    #     'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    #     'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    cmaps['Diverging'] = ['Spectral','Spectral']
    # https: // matplotlib.org / 3.1.0 / tutorials / colors / colormaps.html
    for cmap_category, cmap_list in cmaps.items():
        plot_color_gradients(cmap_category, cmap_list)