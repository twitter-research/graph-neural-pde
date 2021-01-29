import os
import os.path as osp
import argparse
import torch
import numpy as np
import torchvision
from torch_geometric.data import Data, InMemoryDataset, download_url, DataLoader
from torch_geometric.utils import dense_to_sparse
import networkx as nx
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import scipy.sparse as sp
import h5py
from image_opt import get_image_opt


def edge_index_calc(im_height, im_width, im_chan, diags=False):
  edge_list = []

  def oneD():
    for i in range(im_height * im_width):
      # corners
      if i in [0, im_width - 1, im_height * im_width - im_width, im_height * im_width - 1]:
        if i == 0:
          edge_list.append([i, 1])
          edge_list.append([i, im_width])
          edge_list.append([i, im_width + 1]) if diags == True else 0
        elif i == im_width - 1:
          edge_list.append([i, im_width - 2])
          edge_list.append([i, 2 * im_width - 1])
          edge_list.append([i, 2 * im_width - 2]) if diags == True else 0
        elif i == im_height * im_width - im_width:
          edge_list.append([i, im_height * im_width - 2 * im_width])
          edge_list.append([i, im_height * im_width - im_width + 1])
          edge_list.append([i, im_height * im_width - 2 * im_width + 1]) if diags == True else 0
        elif i == im_height * im_width - 1:
          edge_list.append([i, im_height * im_width - 2])
          edge_list.append([i, im_height * im_width - 1 - im_width])
          edge_list.append([i, im_height * im_width - im_width - 2]) if diags == True else 0
      # top edge
      elif i in range(1, im_width - 1):
        edge_list.append([i, i - 1])
        edge_list.append([i, i + 1])
        edge_list.append([i, i + im_width])
        if diags:
          edge_list.append([i, i + im_width - 1])
          edge_list.append([i, i + im_width + 1])
      # bottom edge
      elif i in range(im_height * im_width - im_width, im_height * im_width):
        edge_list.append([i, i - 1])
        edge_list.append([i, i + 1])
        edge_list.append([i, i - im_width])
        if diags:
          edge_list.append([i, i - im_width - 1])
          edge_list.append([i, i - im_width + 1])
      # middle
      else:
        if i % im_width == 0:  # left edge
          edge_list.append([i, i + 1])
          edge_list.append([i, i - im_width])
          edge_list.append([i, i + im_width])
          if diags:
            edge_list.append([i, i - im_width + 1])
            edge_list.append([i, i + im_width + 1])
        elif (i + 1) % im_width == 0:  # right edge
          edge_list.append([i, i - 1])
          edge_list.append([i, i - im_width])
          edge_list.append([i, i + im_width])
          if diags:
            edge_list.append([i, i - im_width - 1])
            edge_list.append([i, i + im_width - 1])
        else:
          edge_list.append([i, i - 1])
          edge_list.append([i, i + 1])
          edge_list.append([i, i - im_width])
          edge_list.append([i, i + im_width])
          if diags:
            edge_list.append([i, i - im_width - 1])
            edge_list.append([i, i - im_width + 1])
            edge_list.append([i, i + im_width - 1])
            edge_list.append([i, i + im_width + 1])
    return edge_list

  edge_list = oneD()
  ret_edge_tensor = torch.tensor(edge_list).T

  # this is wrong need to put colour channels as featurs not extra nodes, saving code in case come back to
  # if im_chan == 1:
  #     edge_list = oneD()
  #     ret_edge_tensor = torch.tensor(edge_list).T
  # else:
  #     edge_list = oneD()
  #     edge_tensor = torch.tensor(edge_list).T
  #     edge_list_3D = []
  #     for i in range(im_chan):
  #         chan_edge_tensor = edge_tensor + i * im_height * im_width
  #         edge_list_3D.append(chan_edge_tensor)
  #     ret_edge_tensor = torch.cat(edge_list_3D,dim=1)
  if diags:
    assert ret_edge_tensor.shape[1] == (8 * (im_width - 2) * (im_height - 2) \
                                        + 2 * 5 * (im_width - 2) + 2 * 5 * (im_height - 2) \
                                        + 4 * 3), "Wrong number of fixed grid edges (inc diags)"
  else:
    assert ret_edge_tensor.shape[1] == (4 * (im_width - 2) * (im_height - 2) \
                                        + 2 * 3 * (im_width - 2) + 2 * 3 * (im_height - 2) \
                                        + 4 * 2), "Wrong number of fixed grid edges (exc diags)"
  return ret_edge_tensor


class ImageInMemory(InMemoryDataset):
  def __init__(self, root, name, opt, type, transform=None, pre_transform=None, pre_filter=None):
    self.name = name
    self.opt = opt
    self.type = type
    super(ImageInMemory, self).__init__(root, transform, pre_transform, pre_filter)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_dir(self):
    return osp.join(self.root, self.name, 'raw')

  @property
  def processed_dir(self):
    return osp.join(self.root, self.name, 'processed')

  @property
  def raw_file_names(self):
    return []

  @property
  def processed_file_names(self):
    return 'data.pt'

  def download(self):
    pass  # download_url(self.url, self.raw_dir)

  def read_data(self):
    if self.opt['im_dataset'] == 'MNIST':
      transform = transforms.Compose([transforms.ToTensor()])#,
                                      # transforms.Normalize((0.1307,), (0.3081,))])
      if self.type == "Train":
        data = torchvision.datasets.MNIST('../data/MNIST/', train=True, download=True,
                                          transform=transform)
      elif self.type == 'Test':
        data = torchvision.datasets.MNIST('../data/MNIST/', train=False, download=True,
                                          transform=transform)
    elif self.opt['im_dataset'] == 'CIFAR':
      # https: // discuss.pytorch.org / t / normalization - in -the - mnist - example / 457 / 7
      transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      if self.type == "Train":
        data = torchvision.datasets.CIFAR10('../data/CIFAR/', train=True, download=True,
                                             transform=transform)
      elif self.type == 'Test':
        data = torchvision.datasets.CIFAR10('../data/CIFAR/', train=False, download=True,
                                            transform=transform)
    return data

  def process(self):
    graph_list = []
    data = self.read_data()
    c, w, h = self.opt['im_chan'], self.opt['im_width'], self.opt['im_height']
    edge_index = edge_index_calc(h, w, c, diags=self.opt['diags'])
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for batch_idx, (data, target) in enumerate(data_loader):
      if self.type == "Train":
        if self.opt['testing_code'] == True and batch_idx > self.opt['train_size'] - 1:
          break
        y = target
      elif self.type == "Test":
        if self.opt['testing_code'] == True and batch_idx > self.opt['test_size'] - 1:
          break
        y = target
      x = data.view(c, w * h)
      x = x.T

      graph = Data(x=x, y=y.unsqueeze(dim=0), edge_index=edge_index)
      graph_list.append(graph)

    # self.data, self.slices = self.collate(graph_list)
    # torch.save((self.data, self.slices), self.processed_paths[0])
    torch.save(self.collate(graph_list), self.processed_paths[0])

class InMemPixelData(ImageInMemory):
  def __init__(self, root, name, opt, type, transform=None, pre_transform=None, pre_filter=None):
    self.name = name
    self.opt = opt
    self.type = type
    super(ImageInMemory, self).__init__(root, transform, pre_transform, pre_filter)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_dir(self):
    return osp.join(self.root, self.name, 'raw')

  @property
  def processed_dir(self):
    return osp.join(self.root, self.name, 'processed')

  @property
  def raw_file_names(self):
    return []

  @property
  def processed_file_names(self):
    return 'data.pt'

  def download(self):
    pass  # download_url(self.url, self.raw_dir)

  def read_data(self):
    if self.opt['im_dataset'] == 'MNIST':
      transform = transforms.Compose([transforms.ToTensor()])  # ,
      # transforms.Normalize((0.1307,), (0.3081,))])
      if self.type == "Train":
        data = torchvision.datasets.MNIST('../data/MNIST/', train=True, download=True,
                                          transform=transform)
      elif self.type == 'Test':
        data = torchvision.datasets.MNIST('../data/MNIST/', train=False, download=True,
                                          transform=transform)
    elif self.opt['im_dataset'] == 'CIFAR':
      # https: // discuss.pytorch.org / t / normalization - in -the - mnist - example / 457 / 7
      transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      if self.type == "Train":
        data = torchvision.datasets.CIFAR10('../data/CIFAR/', train=True, download=True,
                                            transform=transform)
      elif self.type == 'Test':
        data = torchvision.datasets.CIFAR10('../data/CIFAR/', train=False, download=True,
                                            transform=transform)
    return data

  def process(self):
    graph_list = []
    data = self.read_data()
    c, w, h = self.opt['im_chan'], self.opt['im_width'], self.opt['im_height']
    edge_index = edge_index_calc(h, w, c, diags=self.opt['diags'])
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for batch_idx, (data, target) in enumerate(data_loader):
      if self.opt['testing_code'] == True and batch_idx > self.opt['train_size'] - 1:
        break
      x = data.view(c, w * h)
      x = x.T
      # bucket labels in each channel
      pix_labels = []
      for i in range(self.opt['im_chan']):
        y = np.maximum(np.minimum(x * self.opt['pixel_cat'] ,self.opt['pixel_cat']*(0.9999)), 0)
        y = y[:,i]
        y = torch.floor(y).type(torch.LongTensor)
        pix_labels.append(y)
      y = torch.stack(pix_labels, dim=1)
      # y.view(28, 28).detach().numpy()
      full_idx = np.arange(self.opt['num_nodes'])
      # set pixel masks
      rnd_state = np.random.RandomState(seed=12345)
      train_idx = []
      for label in range(y.max().item() + 1):
        # class_idx = np.nonzero(y == label)[:,0]
        class_idx = (y == label).nonzero()[:,0]
        num_in_class = len(class_idx)
        train_idx.extend(rnd_state.choice(class_idx, num_in_class//2, replace=False))
      test_idx = [i for i in full_idx if i not in train_idx]
      train_mask = torch.zeros(self.opt['num_nodes'], dtype=torch.bool)
      test_mask = torch.zeros(self.opt['num_nodes'], dtype=torch.bool)
      train_mask[train_idx] = 1
      test_mask[test_idx] = 1
      graph = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask)
      graph_list.append(graph)

    torch.save(self.collate(graph_list), self.processed_paths[0])


def load_pixel_data(opt):
  print("loading PyG Pixel Data")
  data_name = opt['im_dataset']
  root = '../data'
  name = f"Pixel{data_name}{str(opt['train_size'])}_{str(opt['pixel_cat'])}Cat"
  root = f"{root}/{name}"
  PixelData = InMemPixelData(root, name, opt, type="Train", transform=None, pre_transform=None, pre_filter=None)
  return PixelData


def load_data(opt):
  data_name = opt['im_dataset']
  print("loading PyG in_memory_datasets")
  root = '../data'
  name_train = f"PyG{data_name}{str(opt['train_size'])}Train"
  name_test = f"PyG{data_name}{str(opt['test_size'])}Test"
  root_train = f"{root}/{name_train}"
  root_test = f"{root}/{name_test}"

  PyG_train = ImageInMemory(root_train, name_train, opt, 'Train', transform=None, pre_transform=None, pre_filter=None)
  PyG_test = ImageInMemory(root_test, name_test, opt, 'Test', transform=None, pre_transform=None,
                           pre_filter=None)
  return PyG_train, PyG_test


# from SuperPixData import load_matlab_file, stack_matrices


def create_Superpix75(opt, type, root, processed_file_name=None):
  class IMAGE_IN_MEM(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
      super(IMAGE_IN_MEM, self).__init__(root, transform, pre_transform, pre_filter)
      self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
      return []

    @property
    def processed_file_names(self):
      return [processed_file_name]

    def download(self):
      pass  # download_url(self.url, self.raw_dir)

    def load_labels(self, fname):
      tmp = load_matlab_file(fname, 'labels')
      tmp = tmp.astype(np.int32)
      return tmp.flatten()

    def process(self):
      graph_list = []
      # load
      n_supPix = 75
      path_main = '../data/SuperMNIST/MNIST/'

      if type == "GNN":
        y = torch.tensor(opt['num_class'] - 1)
      elif type == "Train":
        path_train_vals = os.path.join(path_main, 'datasets/mnist_superpixels_data_%d/train_vals.mat' % n_supPix)
        path_coords_train = os.path.join(path_main,
                                         'datasets/mnist_superpixels_data_%d/train_patch_coords.mat' % n_supPix)
        path_train_labels = os.path.join(path_main, 'datasets/MNIST_preproc_train_labels/MNIST_labels.mat')
        path_train_centroids = os.path.join(path_main,
                                            'datasets/mnist_superpixels_data_%d/train_centroids.mat' % n_supPix)

        vals_train = load_matlab_file(path_train_vals, 'vals')
        tmp = load_matlab_file(path_coords_train, 'patch_coords')
        coords_train = stack_matrices(tmp, n_supPix)

        # compute the adjacency matrix
        adj_mat_train = np.zeros(
          (coords_train.shape[0], coords_train.shape[1], coords_train.shape[2]))
        for k in range(coords_train.shape[0]):
          adj_mat_train[k, :, :] = np.isfinite(coords_train[k, :, :, 1])

        # data labels [0,...,9]
        train_labels = self.load_labels(path_train_labels)

        # what are these? one per datapoint
        idx_centroids_train = load_matlab_file(path_train_centroids, 'idx_centroids')

        batch_size = opt['batch_size']
        for i in range(opt['train_size'] // batch_size):
          x = vals_train[i * batch_size:(i + 1) * batch_size + 1, :]
          y = train_labels[i]
          pos = coords_train

          edge_index, edge_attr = dense_to_sparse(adj_mat_train[i, :, :].squeeze())

          graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos))

      elif type == "Test":
        path_test_vals = os.path.join(path_main, 'datasets/mnist_superpixels_data_%d/test_vals.mat' % n_supPix),
        path_coords_test = os.path.join(path_main,
                                        'datasets/mnist_superpixels_data_%d/test_patch_coords.mat' % n_supPix),
        path_test_labels = os.path.join(path_main, 'datasets/MNIST_preproc_test_labels/MNIST_labels.mat'),
        path_test_centroids = os.path.join(path_main,
                                           'datasets/mnist_superpixels_data_%d/test_centroids.mat' % n_supPix)

        vals_test = load_matlab_file(path_test_vals, 'vals')
        tmp = load_matlab_file(path_coords_test, 'patch_coords')
        coords_test = stack_matrices(tmp, n_supPix)

        # compute the adjacency matrix
        adj_mat_test = np.zeros(
          (coords_test.shape[0], coords_test.shape[1], coords_test.shape[2]))
        for k in range(coords_test.shape[0]):
          adj_mat_test[k, :, :] = np.isfinite(coords_test[k, :, :, 1])

        test_labels = self.load_labels(path_test_labels)

      self.data, self.slices = self.collate(graph_list)
      torch.save((self.data, self.slices), self.processed_paths[0])

  return IMAGE_IN_MEM(root)


def load_Superpix75Mat(opt):  # , path='../data/SuperMNIST/MNIST/datasets/mnist_superpixels_data_75/'):

  # 'path_train_vals' : os.path.join(path_main, 'datasets/mnist_superpixels_data_%d/train_vals.mat' % n_supPix),
  # 'path_coords_train' : os.path.join(path_main, 'datasets/mnist_superpixels_data_%d/train_patch_coords.mat' % n_supPix),
  # 'path_train_labels' : os.path.join(path_main, 'datasets/MNIST_preproc_train_labels/MNIST_labels.mat'),
  # 'path_train_centroids' : os.path.join(path_main, 'datasets/mnist_superpixels_data_%d/train_centroids.mat' % n_supPix)},
  # 'test':{
  # 'path_test_vals' : os.path.join(path_main, 'datasets/mnist_superpixels_data_%d/test_vals.mat' % n_supPix),
  # 'path_coords_test' : os.path.join(path_main, 'datasets/mnist_superpixels_data_%d/test_patch_coords.mat' % n_supPix),
  # 'path_test_labels' : os.path.join(path_main, 'datasets/MNIST_preproc_test_labels/MNIST_labels.mat'),
  # 'path_test_centroids' : os.path.join(path_main, 'datasets/mnist_superpixels_data_%d/test_centroids.mat' % n_supPix)}}

  print("creating in_memory_datasets")
  # type = "GNN"
  # Graph_GNN = create_Superpix75(opt, type,
  #             root='../data/SuperPix75'+'_'+type+'/', processed_file_name='GraphSuperPix75'+type+'.pt')
  type = "Train"
  Graph_train = create_Superpix75(opt, type,
                                  root='../data/SuperPix75' + str(opt['train_size']) + type + '/',
                                  processed_file_name='GraphSuperPix75' + str(opt['train_size']) + type + '.pt')
  type = "Test"
  Graph_test = create_Superpix75(opt, type,
                                 root='../data/SuperPix75' + str(opt['test_size']) + type + '/',
                                 processed_file_name='GraphSuperPix75' + str(opt['test_size']) + type + '.pt')

  return Graph_train, Graph_test


def imshow(img):
  img = img / 2 + 0.5  # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  args = parser.parse_args()
  opt = vars(args)
  opt = get_image_opt(opt)
  load_data(opt)
  # data = torch.load('data/PyGMNIST128Train/processed/data.pt')

  # load_Superpix75Mat(opt)

  # Cora = get_dataset('Cora', '../data', False)
  # gnn = GNN(self.opt, dataset, device=self.device)
  # odeblock = gnn.odeblock
  # func = odeblock.odefunc

  # img_size = 32#28
  # im_width = img_size
  # im_height = img_size
  # im_chan = 3 #1
  # exdataset = 'CIFAR' #'MNIST'

  # train_loader = torch.utils.data.DataLoader(
  #   torchvision.datasets.MNIST('data/' + exdataset + '/', train=True, download=True,
  #                              transform=torchvision.transforms.Compose([
  #                                torchvision.transforms.ToTensor(),
  #                                torchvision.transforms.Normalize(
  #                                  (0.1307,), (0.3081,))
  #                              ])),
  #                                 batch_size=1, shuffle=True)

  # transform = transforms.Compose(
  #     [transforms.ToTensor(),
  #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  #
  # train_loader = torch.utils.data.DataLoader(
  #   torchvision.datasets.CIFAR10('data/' + exdataset + '/', train=True, download=True,
  #                              transform=transform),
  #                                 batch_size=1, shuffle=True)
  #
  #
  # edge_index = edge_index_calc(im_height, im_width)
  #
  # opt = get_image_opt({})
  # Graph = create_in_memory_dataset(opt, "Train", train_loader, edge_index, im_height, im_width, im_chan,
  #                                                                         root='./data/Graph' + exdataset + 'GNN/',
  #                                                                         processed_file_name='Graph' + exdataset + 'GNN2.pt')
  #
  # fig = plt.figure(figsize=(32,62))
  # # for i in range(6):
  # #     plt.subplot(2, 3, i + 1)
  #
  # for i in range(20):
  #     plt.subplot(5, 4, i + 1)
  #     plt.tight_layout()
  #     digit = Graph[i]
  #     plt.title("Ground Truth: {}".format(digit.y.item()))
  #     plt.xticks([])
  #     plt.yticks([])
  #     A = digit.x#.view(im_height, im_width, im_chan)
  #     A = A.numpy()
  #     A = np.reshape(A, (im_height, im_width, im_chan), order='F')
  #     A = A / 2 + 0.5  # unnormalize
  #     plt.imshow(np.transpose(A, (1, 0, 2)))
  # # plt.show()
  # plt.savefig("GraphImages.png", format="PNG")
