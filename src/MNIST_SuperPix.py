import argparse
import torch
import torchvision
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import skimage
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage import segmentation
import torchvision.transforms as transforms

def calc_centroids(pixel_lables, num_centroids):
    centroids = {}
    for i in range(num_centroids):
        indices = np.where(pixel_lables==i)
        pixel_lables[pixel_lables==i]
        x_av = np.mean(indices[1])  #x is im_width is 1st axis
        y_av = np.mean(indices[0])  #y is im_height is 0th axis
        centroids[i] = (x_av, y_av)
    return centroids


def get_centroid_coords_array(num_centroids, centroids):
    x_coords = []
    y_coords = []
    for i in range(num_centroids):
        x_coords.append(centroids[i][1]) #x is im_width is 1st axis
        y_coords.append(centroids[i][0]) #y is im_height is 0th axis
    return x_coords, y_coords


def find_neighbours(labels, boundaries, im_height, im_width):
    """return a dictionary of centroid neighbours {centroid:{neighbours}}"""
    neighbour_dict = {}
    for i in range(im_height):
        for j in range(im_width):
            pix = i * im_width + j
            if boundaries[i][j] ==1:
                neighbours = np.array([pix-im_width,pix-1,pix+1,pix+im_width])
                neighbours = neighbours[neighbours > 0]
                neighbours = neighbours[neighbours < im_height*im_width]
                neighbours = neighbours[np.logical_or(neighbours//im_width == pix//im_width, neighbours%im_width == pix%im_width)]
                for n in neighbours:
                    if labels[pix // im_width][pix % im_width] in neighbour_dict:
                        temp_set = neighbour_dict[labels[pix//im_width][pix%im_width]]
                        temp_set.add(labels[n//im_width][n%im_width])
                        neighbour_dict[labels[pix//im_width][pix%im_width]] = temp_set
                    else:
                        neighbour_dict[labels[pix//im_width][pix%im_width]] = {labels[n//im_width][n%im_width]}
    return neighbour_dict


def calc_centroid_labels(num_centroids, pixel_lables, pixel_values):
    """for each centroid extract its value from its original pixel lables"""
    #TODO needs a rethink
    x = []
    for c in range(num_centroids):
        CL = pixel_values[np.ix_(np.where(pixel_lables == c)[0], np.where(pixel_lables == c)[1])]

        if np.all(np.amax(CL, axis=(0,1)) == np.amin(CL, axis=(0,1))):
            x.append(np.amax(CL, axis=(0,1)))
        else:
            #todo why aren't patch values the same?
            print(c)
            print(np.amax(CL, axis=(0, 1)))
            print(np.amin(CL, axis=(0, 1)))
            x.append(np.amax(CL, axis=(0, 1)))
    return np.array(x)


def create_edge_index_from_neighbours(neighbour_dict):
    edge_index = [[],[]]
    for i, J in neighbour_dict.items():
        edge_index[0].extend([i]*len(J))
        edge_index[1].extend(list(J))
    return torch.tensor(edge_index,dtype=int)

def transform_objects(im_height, im_width, ySF, xSF, pixel_values, pixel_labels, centroids):
    #applying transform to various objects
    resized_pixel_values = skimage.transform.resize(pixel_values, (im_height*ySF,im_width*xSF,3),
                               mode='edge',
                               anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               preserve_range=True,
                               order=0)

    resized_pixel_labels = skimage.transform.resize(pixel_labels, (im_height*ySF,im_width*xSF),
                               mode='edge',
                               anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               preserve_range=True,
                               order=0)
    resized_centroids = {key:((pos[0]+1/2)*xSF, (pos[1]+1/2)*ySF) for key, pos in centroids.items()}
    #don't need to do the x/y reversal here as already performed

    return resized_pixel_values, resized_pixel_labels, resized_centroids

def read_data(im_dataset, type):
    if im_dataset == 'MNIST':
      transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
      if type == "Train":
        data = torchvision.datasets.MNIST('../data/MNIST/', train=True, download=True,
                                          transform=transform)
      elif type == 'Test':
        data = torchvision.datasets.MNIST('../data/MNIST/', train=False, download=True,
                                          transform=transform)
    elif im_dataset == 'CIFAR':
      transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      if type == "Train":
        data = torchvision.datasets.CIFAR10('../data/CIFAR/', train=True, download=True,
                                             transform=transform)
      elif type == 'Test':
        data = torchvision.datasets.CIFAR10('../data/CIFAR/', train=False, download=True,
                                            transform=transform)
    return data

def main(opt):
    ###coordinate convention
    # X is IM_WIDTH is left to right is in the 1st axis of the matrix
    im_width = opt['im_width']
    # Y is IM_HEIGHT is top to bottom is in the 0th axis of the matrix
    im_height = opt['im_height']

    # torch.manual_seed(1)
    data_loader = torch.utils.data.DataLoader(read_data(opt['im_dataset'], "Train"), batch_size=32, shuffle=True)
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # PLOT MNIST EXAMPLES
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    img = example_data[0][0].numpy()
    img = img - img.min()
    img = img / img.max() * 255.0
    img = img.astype(np.double)
    # For the superpixel segmentation, we use SLIC from skimage.

    multichannel = opt['im_chan'] > 1
    pixel_labels = slic(img, n_segments=75, multichannel=multichannel)  # Segments image using k-means clustering
    pixel_values = label2rgb(pixel_labels, bg_label=0)

    # PLOT COLOURED PATCHES WITH CENTROIDS
    fig, ax = plt.subplots()
    # plt.axis('off')
    plt.imshow(pixel_values)
    pixel_lables = pixel_labels
    num_centroids = np.max(pixel_lables) + 1  # required to add 1 for full coverage
    centroids = calc_centroids(pixel_labels, num_centroids)
    y_coords, x_coords = get_centroid_coords_array(num_centroids, centroids)
    ax.scatter(x=x_coords, y=y_coords)
    for i in range(num_centroids):
        ax.annotate(i, (x_coords[i], y_coords[i]))
    plt.show()  # PLOT COLOURED PATCHES WITH CENTROIDS

    # PLOT B/W PATCHES WITHOUT CENTROIDS OR SEGMENTATION
    color2 = pixel_values.copy()
    for entry in np.unique(pixel_labels):
        color2[np.where(pixel_labels == entry)[0], np.where(pixel_labels == entry)[1]] = \
            np.max(img[np.where(pixel_labels == entry)[0], np.where(pixel_labels == entry)[1]])
    color2 = color2 - color2.min()
    color2 = color2 / color2.max() * 1.0
    color2 = color2.astype(np.double)
    plt.axis('off')
    plt.imshow(color2)
    plt.show()  # PLOT B/W PATCHES WITHOUT CENTROIDS OR SEGMENTATION

    # RESIZING - (TEST IF NEEDED)
    SF = 480  # 480    #for some reason this is fixed
    heightSF = SF / im_height
    widthSF = SF / im_width
    pixel_values = color2
    r_pixel_values, r_pixel_labels, r_centroids = transform_objects(im_height, im_width, heightSF, widthSF,
                                                                    pixel_values, pixel_labels, centroids)
    r_y_coords, r_x_coords = get_centroid_coords_array(num_centroids, r_centroids)

    # SEGMENTATION
    r_pixel_labels = r_pixel_labels.astype(np.int)
    out = segmentation.mark_boundaries(r_pixel_values, r_pixel_labels, (1, 0, 0))
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(out)
    # centroid labels
    for i in range(num_centroids):
        ax.annotate(i, (r_x_coords[i], r_y_coords[i]), c="red")
    ax.scatter(x=r_x_coords, y=r_y_coords)
    plt.show()  # SEGMENTATION

    # CONSTRUCTING PyG
    boundaries = skimage.segmentation.find_boundaries(pixel_labels, mode='inner').astype(np.uint8)
    neighbour_dict = find_neighbours(pixel_labels, boundaries, im_height=28, im_width=28)
    edge_index = create_edge_index_from_neighbours(neighbour_dict)
    x = calc_centroid_labels(num_centroids, pixel_lables, pixel_values=pixel_values)
    x = torch.tensor(x)
    pos_centroids = torch.tensor([centroids[i] for i in range(num_centroids)])
    graph = Data(x=x, edge_index=edge_index, pos=pos_centroids, orig_image=example_data[0][0], y=example_targets[0])

    # PLOT NXGRAPH  #TODO this is upside down
    NXgraph = to_networkx(graph)
    fig, (ax, cbar_ax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [1, 0.05]})  # figsize=(4, 4),
    nx.draw(NXgraph, centroids, ax=ax, node_size=300 / 4, edge_color="lime",
            node_color=x.sum(axis=1, keepdim=False), cmap=plt.get_cmap('Spectral'))
    ax.set_xlabel("Centroid Graph")
    fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('Spectral')),
                 cax=cbar_ax, orientation="horizontal")
    # https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    # https: // stackoverflow.com / questions / 13310594 / positioning - the - colorbar
    plt.show()  # PLOT NXGRAPH

    # FINAL IMAGE
    fig, ax = plt.subplots()
    # fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.05]})
    ax.axis('off'), cbar_ax.axis('off')
    ax.imshow(out)
    for i in range(num_centroids):
        ax.annotate(i, (r_x_coords[i], r_y_coords[i]), c="red")
    ax.scatter(x=r_x_coords, y=r_y_coords)
    nx.draw(NXgraph, r_centroids, ax=ax, node_size=300 / 4, edge_color="lime",
            node_color=x.sum(axis=1, keepdim=False), cmap=plt.get_cmap('Spectral'))
    # fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('Spectral')),
    #              cax=cbar_ax, orientation="vertical")
    plt.show()  # FINAL IMAGE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--use_cora_defaults', action='store_true',
    #                     help='Whether to run with best params for cora. Overrides the choice of dataset')
    # parser.add_argument('--dataset', type=str, default='Cora',
    #                     help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--alpha_sigmoid', type=bool, default=True, help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
    # ODE args
    parser.add_argument('--method', type=str, default='dopri5',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument(
        "--adjoint_method", type=str, default="adaptive_heun",
        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
    )
    parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument('--simple', type=bool, default=True,
                        help='If try get rid of alpha param and the beta*x0 source term')
    # SDE args
    parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
    parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
    parser.add_argument('--adaptive', type=bool, default=False, help='use adaptive step sizes')
    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', type=bool, default=False,
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument("--max_nfe", type=int, default=5000, help="Maximum number of function evaluations allowed.")
    parser.add_argument('--reweight_attention', type=bool, default=False,
                        help="multiply attention scores by edge weights before softmax")
    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")
    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")
    # rewiring args
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                        help="obove this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                        help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    # visualisation args
    parser.add_argument('--testing_code', type=bool, default=False, help='run on limited size training/test sets')
    parser.add_argument('--use_image_defaults', default='MNIST', help='sets as per function get_image_opt')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_size', type=int, default=128, help='Batch size')
    parser.add_argument('--test_size', type=int, default=128, help='Batch size')
    parser.add_argument('--batched', type=bool, default=True, help='Batching')
    parser.add_argument('--im_width', type=int, default=28, help='im_width')
    parser.add_argument('--im_height', type=int, default=28, help='im_height')
    parser.add_argument('--im_chan', type=int, default=1, help='im_height')
    parser.add_argument('--num_class', type=int, default=10, help='im_height')
    parser.add_argument('--diags', type=bool, default=False, help='Edge index include diagonal diffusion')
    parser.add_argument('--im_dataset', type=str, default='MNIST', help='MNIST, CIFAR')
    parser.add_argument('--num_nodes', type=int, default=28 ** 2, help='im_width')

    args = parser.parse_args()
    opt = vars(args)
    main(opt)
