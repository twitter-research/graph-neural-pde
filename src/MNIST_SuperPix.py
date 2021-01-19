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

if __name__ == "__main__":
    ###coordinate convention
    #X is IM_WIDTH is left to right is in the 1st axis of the matrix
    #Y is IM_HEIGHT is top to bottom is in the 0th axis of the matrix
    im_height = 28
    im_width = 28

    # torch.manual_seed(1)
    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('../data/MNIST/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=32, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # PLOT MNIST EXAMPLES
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
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
    pixel_labels = slic(img, n_segments=75) #Segments image using k-means clustering
    pixel_values = label2rgb(pixel_labels, bg_label=0)

    # PLOT COLOURED PATCHES WITH CENTROIDS
    fig, ax = plt.subplots()
    # plt.axis('off')
    plt.imshow(pixel_values)
    pixel_lables = pixel_labels
    num_centroids = np.max(pixel_lables)+1 #required to add 1 for full coverage
    centroids = calc_centroids(pixel_labels, num_centroids)
    y_coords, x_coords = get_centroid_coords_array(num_centroids, centroids)
    ax.scatter(x=x_coords, y=y_coords)
    for i in range(num_centroids):
        ax.annotate(i, (x_coords[i], y_coords[i]))
    plt.show()   # PLOT COLOURED PATCHES WITH CENTROIDS

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
    plt.show() #PLOT B/W PATCHES WITHOUT CENTROIDS OR SEGMENTATION

    #RESIZING - (TEST IF NEEDED)
    SF = 480 #480    #for some reason this is fixed
    heightSF = SF/ im_height
    widthSF = SF / im_width
    pixel_values = color2
    r_pixel_values, r_pixel_labels, r_centroids = transform_objects(im_height, im_width, heightSF, widthSF,
                                                              pixel_values, pixel_labels, centroids)
    r_y_coords, r_x_coords = get_centroid_coords_array(num_centroids, r_centroids)

    #SEGMENTATION
    r_pixel_labels = r_pixel_labels.astype(np.int)
    out = segmentation.mark_boundaries(r_pixel_values, r_pixel_labels, (1, 0, 0))
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(out)
    #centroid labels
    for i in range(num_centroids):
        ax.annotate(i, (r_x_coords[i], r_y_coords[i]),c="red")
    ax.scatter(x=r_x_coords, y=r_y_coords)
    plt.show()#SEGMENTATION

    #CONSTRUCTING PyG
    boundaries = skimage.segmentation.find_boundaries(pixel_labels, mode='inner').astype(np.uint8)
    neighbour_dict = find_neighbours(pixel_labels, boundaries, im_height=28, im_width=28)
    edge_index = create_edge_index_from_neighbours(neighbour_dict)
    x = calc_centroid_labels(num_centroids, pixel_lables, pixel_values=pixel_values)
    x = torch.tensor(x)
    pos_centroids = torch.tensor([centroids[i] for i in range(num_centroids)])
    graph = Data(x=x, edge_index=edge_index, pos=pos_centroids, orig_image=example_data[0][0], y=example_targets[0])

    #PLOT NXGRAPH  #TODO this is upside down
    NXgraph = to_networkx(graph)
    fig, (ax, cbar_ax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [1, 0.05]}) #figsize=(4, 4),
    nx.draw(NXgraph, centroids, ax=ax, node_size=300/4,edge_color="lime",
            node_color=x.sum(axis=1,keepdim=False), cmap=plt.get_cmap('Spectral'))
    ax.set_xlabel("Centroid Graph")
    fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('Spectral')),
                 cax=cbar_ax, orientation="horizontal")
    # https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    # https: // stackoverflow.com / questions / 13310594 / positioning - the - colorbar
    plt.show()#PLOT NXGRAPH

    #FINAL IMAGE
    fig, ax = plt.subplots()
    # fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.05]})
    ax.axis('off'), cbar_ax.axis('off')
    ax.imshow(out)
    for i in range(num_centroids):
        ax.annotate(i, (r_x_coords[i], r_y_coords[i]),c="red")
    ax.scatter(x=r_x_coords, y=r_y_coords)
    nx.draw(NXgraph, r_centroids, ax=ax, node_size=300/4,edge_color="lime",
            node_color=x.sum(axis=1,keepdim=False), cmap=plt.get_cmap('Spectral'))
    # fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('Spectral')),
    #              cax=cbar_ax, orientation="vertical")
    plt.show() #FINAL IMAGE