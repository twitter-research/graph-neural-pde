import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

def calc_centroids(pixel_lables, num_centroids):
    centroids = {}

    for i in range(num_centroids):
        indices = np.where(pixel_lables==i)
        pixel_lables[pixel_lables==i]
        # x_coords = indices % im_width
        # y_coords = indices // im_width
        vertical_av = np.mean(indices[0])
        horizontal_av = np.mean(indices[1])
        centroids[i] = (vertical_av, horizontal_av)
    return centroids


def find_neighbours(labels, boundaries, im_height, im_width):
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
    x = []
    for c in range(num_centroids):
        # pixel_lables[pixel_lables==c]
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

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# For the superpixel segmentation, we use SLIC from skimage.
import skimage
from skimage.segmentation import slic
from skimage.color import label2rgb

img = example_data[0][0].numpy()
img = img - img.min()
img = img / img.max() * 255.0
img = img.astype(np.double)
s = slic(img, n_segments=75) #Segments image using k-means clustering
color1 = label2rgb(s, bg_label=0)

fig, ax = plt.subplots()
# plt.axis('off')
plt.imshow(color1)
pixel_lables = s
num_centroids = np.max(pixel_lables)+1 #seems to add 1
centroids = calc_centroids(s, num_centroids)
vertical = []
horizontal = []
for i in range(num_centroids):
    vertical.append(centroids[i][0])
    horizontal.append(centroids[i][1])

# [x.append(centroids[i][0]) and y.append(centroids[i][1]) for i in range(num_centroids)]
# ax.scatter(x=horizontal, y=vertical)
for i in range(num_centroids):
    ax.annotate(i, (horizontal[i], vertical[i]))
ax.scatter(x=horizontal, y=vertical)
plt.show()

##black and white version
color2 = color1.copy()
for entry in np.unique(s):
    color2[np.where(s==entry)[0], np.where(s==entry)[1]] = \
      np.max(img[np.where(s==entry)[0], np.where(s==entry)[1]])

color2 = color2 - color2.min()
color2 = color2 / color2.max() * 1.0
color2 = color2.astype(np.double)

plt.axis('off')
plt.imshow(color2)
plt.show()
# Let us visualize the patch borders:
from skimage import segmentation

c2 = skimage.transform.resize(color2, (480,480,3),
                               mode='edge',
                               anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               preserve_range=True,
                               order=0)

s2 = skimage.transform.resize(s, (480,480),
                               mode='edge',
                               anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               preserve_range=True,
                               order=0)

s2 = s2.astype(np.int)
out = segmentation.mark_boundaries(c2, s2, (1, 0, 0))
fig, ax = plt.subplots()
plt.axis('off')
ax.imshow(out)
horizontal = np.array(horizontal) * 480 / 28 + 480 / (28 * 2)
vertical = np.array(vertical) * 480 / 28 + 480 / (28 * 2)

for i in range(num_centroids):
    ax.annotate(i, (horizontal[i], vertical[i]),c="red")
ax.scatter(x=horizontal, y=vertical)
plt.show()



boundaries = skimage.segmentation.find_boundaries(s, mode='inner').astype(np.uint8)



neighbour_dict = find_neighbours(s, boundaries, im_height=28, im_width=28)

edge_index = create_edge_index_from_neighbours(neighbour_dict)


x = calc_centroid_labels(num_centroids, pixel_lables, pixel_values=color2)
x = torch.tensor(x)
pos_centroids = torch.tensor([centroids[i] for i in range(num_centroids)])
graph = Data(x=x, edge_index=edge_index, pos=pos_centroids)

NXgraph = to_networkx(graph)
nx.draw(NXgraph, centroids,node_size=300/4,edge_color="lime")
plt.show()

######
fig, ax = plt.subplots()
plt.axis('off')
ax.imshow(out)

resized_centroids = {key:(pos[1]*480/28+480/(28*2),pos[0]*480/28+480/(28*2)) for key, pos in centroids.items()}

for i in range(num_centroids):
    ax.annotate(i, (horizontal[i], vertical[i]),c="red")
ax.scatter(x=horizontal, y=vertical)
nx.draw(NXgraph, resized_centroids,node_size=300/4,edge_color="lime")
plt.show()