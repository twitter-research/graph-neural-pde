import os
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

def get_image_opt(opt):
  opt['im_dataset'] = 'MNIST'  #datasets = ['MNIST','CIFAR']

  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555  #### 0?
  opt['alpha'] = 0.918
  opt['time'] = 12.1
  opt['augment'] = False #True   #False need to view image
  opt['attention_dropout'] = 0
  opt['adjoint'] = False

  opt['epoch'] = 4 #3 #10#20 #400
  opt['batch_size'] = 64 #64  # doing batch size for mnist
  opt['train_size'] = 512 #128#512 #4096 #2048 #512 #2047:#4095:#511:#5119:#1279:#
  opt['test_size'] = 64 #2559:#1279:
  assert (opt['train_size']) % opt['batch_size'] == 0, "train_size needs to be multiple of batch_size"
  assert (opt['test_size']) % opt['batch_size'] == 0, "test_size needs to be multiple of batch_size"


  if opt['im_dataset'] == 'MNIST':
    opt['im_width'] = 28
    opt['im_height'] = 28
    opt['im_chan'] = 1
    opt['hidden_dim'] = 1 #16    #### 1 or 3 rgb?
    opt['num_feature'] = 1  # 1433   #### 1 or 3 rgb?
    opt['num_class'] = 10  # 7  #### mnist digits

  elif opt['im_dataset'] == 'CIFAR':
    opt['im_width'] = 32
    opt['im_height'] = 32
    opt['im_chan'] = 3
    # ????
    opt['hidden_dim'] = 3 #16    #### 1 or 3 rgb?
    opt['num_feature'] = 3  # 1433   #### 1 or 3 rgb?
    opt['num_class'] = 10  # 7  #### mnist digits

  opt['num_nodes'] = opt['im_height'] * opt['im_width'] * opt['im_chan'] #2708  ###pixels
  opt['simple'] = False #True
  opt['diags'] = True
  opt['ode'] = 'ode' #'att' don't think att is implmented properly on this codebase?
  opt['linear_attention'] = True
  opt['batched'] = True
  return opt


def edge_index_calc(im_height, im_width, im_chan, diags = False):
    edge_list = []
    def oneD():
        for i in range(im_height * im_width):
            #corners
            if i in [0, im_width-1, im_height * im_width - im_width, im_height * im_width - 1]:
                if i  == 0:
                    edge_list.append([i,1])
                    edge_list.append([i,im_width])
                    edge_list.append([i,im_width + 1]) if diags == True else 0
                elif i == im_width - 1:
                    edge_list.append([i, im_width - 2])
                    edge_list.append([i, 2 * im_width - 1])
                    edge_list.append([i, 2 * im_width - 2]) if diags == True else 0
                elif i == im_height * im_width - im_width:
                    edge_list.append([i, im_height * im_width - 2*im_width])
                    edge_list.append([i, im_height * im_width - im_width+1])
                    edge_list.append([i, im_height * im_width - 2*im_width+1]) if diags == True else 0
                elif i == im_height * im_width - 1:
                    edge_list.append([i, im_height * im_width - 2])
                    edge_list.append([i, im_height * im_width - 1 - im_width])
                    edge_list.append([i, im_height * im_width - im_width - 2]) if diags == True else 0
            # top edge
            elif i in range(1,im_width-1):
                edge_list.append([i,i-1])
                edge_list.append([i,i+1])
                edge_list.append([i,i+im_width])
                if diags:
                    edge_list.append([i, i + im_width -1])
                    edge_list.append([i, i + im_width + 1])
            # bottom edge
            elif i in range(im_height * im_width - im_width, im_height * im_width):
                edge_list.append([i,i-1])
                edge_list.append([i,i+1])
                edge_list.append([i,i-im_width])
                if diags:
                    edge_list.append([i, i - im_width -1])
                    edge_list.append([i, i - im_width + 1])
            # middle
            else:
                if i % im_width == 0: # left edge
                    edge_list.append([i,i+1])
                    edge_list.append([i,i-im_width])
                    edge_list.append([i,i+im_width])
                    if diags:
                        edge_list.append([i, i - im_width + 1])
                        edge_list.append([i, i + im_width + 1])
                elif (i + 1) % im_width == 0: # right edge
                    edge_list.append([i,i-1])
                    edge_list.append([i,i-im_width])
                    edge_list.append([i,i+im_width])
                    if diags:
                        edge_list.append([i, i - im_width - 1])
                        edge_list.append([i, i + im_width - 1])
                else:
                    edge_list.append([i,i-1])
                    edge_list.append([i,i+1])
                    edge_list.append([i,i-im_width])
                    edge_list.append([i,i+im_width])
                    if diags:
                        edge_list.append([i, i - im_width - 1])
                        edge_list.append([i, i - im_width + 1])
                        edge_list.append([i, i + im_width - 1])
                        edge_list.append([i, i + im_width + 1])
        return edge_list

    edge_list = oneD()
    ret_edge_tensor = torch.tensor(edge_list).T

    #this is wrong need to put colour channels as featurs not extra nodes, saving code in case come back to
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
        assert ret_edge_tensor.shape[1] == (8*(im_width-2)*(im_height-2)\
                                    + 2*5*(im_width-2) + 2*5*(im_height-2)\
                                    + 4*3) ,"Wrong number of fixed grid edges (inc diags)"
    else:
        assert ret_edge_tensor.shape[1] == (4*(im_width-2)*(im_height-2) \
                                    + 2*3*(im_width-2) + 2*3*(im_height-2)\
                                    + 4*2) ,"Wrong number of fixed grid edges (exc diags)"
    return ret_edge_tensor


def create_in_memory_dataset(opt, type, data_loader, edge_index, im_height, im_width, im_chan, root, processed_file_name=None):
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
            pass #download_url(self.url, self.raw_dir)

        # @property
        # def num_classes(self):
        #     r"""The number of classes in the dataset."""
        #     y = self.data.y
        #     return y.max().item() + 1 if y.dim() == 1 else y.size(1)
        def process(self):
            graph_list = []
            for batch_idx, (data, target) in enumerate(data_loader):
                if type == "GNN":
                    if batch_idx > 0:
                        break
                    self.tensor = torch.tensor(opt['num_class'] - 1)
                    y = self.tensor  #<- hack the datset num_classes property (code above)
                elif type == "Train":
                    if opt['testing_code'] == True and batch_idx > opt['train_size'] - 1:
                        break
                    y = target
                elif type == "Test":
                    if opt['testing_code'] == True and batch_idx > opt['test_size'] - 1:
                        break
                    y = target
                x = data.view(im_chan, im_width * im_height)#, -1)
                x = x.T

                graph = Data(x=x, y=y.unsqueeze(dim=0), edge_index=edge_index)
                graph_list.append(graph)

            self.data, self.slices = self.collate(graph_list)
            torch.save((self.data, self.slices), self.processed_paths[0])

    return IMAGE_IN_MEM(root)

def load_data(opt):
  im_height = opt['im_height']
  im_width = opt['im_width']
  im_chan = opt['im_chan']
  exdataset = opt['im_dataset']

  if opt['im_dataset'] == 'MNIST':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    data_train = torchvision.datasets.MNIST('../data/'+exdataset + '/', train=True, download=True,
                                            transform=transform)
    data_test = torchvision.datasets.MNIST('../data/'+exdataset+'/', train=False, download=True,
                                           transform=transform)
  elif opt['im_dataset'] == 'CIFAR':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_train = torchvision.datasets.CIFAR10('../data/' + exdataset + '/', train=True, download=True,
                                            transform=transform)
    data_test = torchvision.datasets.CIFAR10('../data/' + exdataset + '/', train=False, download=True,
                                           transform=transform)

  train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)
  test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=True)

  edge_index = edge_index_calc(im_height, im_width, im_chan, diags=opt['diags'])
  print("creating in_memory_datasets")
  if opt['testing_code'] == True:
      Graph_GNN = create_in_memory_dataset(opt, "GNN", train_loader, edge_index, im_height, im_width, im_chan,
                                             root='../data/PyG'+exdataset+'GNN/', processed_file_name='Graph'+exdataset+'GNN.pt')
      Graph_train = create_in_memory_dataset(opt, "Train", train_loader, edge_index, im_height, im_width, im_chan,
                                             root='../data/PyG'+exdataset+ str(opt['train_size'])+'Train/', processed_file_name='Graph'+exdataset+str(opt['train_size']) +'Train.pt')
      Graph_test = create_in_memory_dataset(opt, "Test", test_loader, edge_index, im_height, im_width, im_chan,
                                             root='../data/PyG'+exdataset+ str(opt['test_size'])+'Test/', processed_file_name='Graph'+exdataset+str(opt['test_size']) +'Test.pt')
  else:
      Graph_GNN = create_in_memory_dataset(opt, "GNN", train_loader, edge_index, im_height, im_width, im_chan,
                                             root='../data/PyG'+exdataset+'GNN/', processed_file_name='PyG'+exdataset+'GNN.pt')
      Graph_train = create_in_memory_dataset(opt, "Train", train_loader, edge_index, im_height, im_width, im_chan,
                                             root='../data/PyG'+exdataset+'Train/', processed_file_name='PyG'+exdataset+'Train.pt')
      Graph_test = create_in_memory_dataset(opt, "Test", test_loader, edge_index, im_height, im_width, im_chan,
                                             root='../data/PyG'+exdataset+'Test/', processed_file_name='PyG'+exdataset+'Test.pt')
  return Graph_GNN, Graph_train, Graph_test

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
            pass #download_url(self.url, self.raw_dir)

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
                y = torch.tensor(opt['num_class']-1)
            elif type == "Train":
                path_train_vals = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/train_vals.mat' % n_supPix)
                path_coords_train = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/train_patch_coords.mat' % n_supPix)
                path_train_labels = os.path.join(path_main, 'datasets/MNIST_preproc_train_labels/MNIST_labels.mat')
                path_train_centroids = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/train_centroids.mat' % n_supPix)

                vals_train = load_matlab_file(path_train_vals, 'vals')
                tmp = load_matlab_file(path_coords_train, 'patch_coords')
                coords_train = stack_matrices(tmp, n_supPix)

                # compute the adjacency matrix
                adj_mat_train = np.zeros(
                    (coords_train.shape[0], coords_train.shape[1], coords_train.shape[2]))
                for k in range(coords_train.shape[0]):
                    adj_mat_train[k, :, :] = np.isfinite(coords_train[k, :, :, 1])

                train_labels = self.load_labels(path_train_labels)
                batch_size = opt['batch_size']
                for i in range(opt['train_size'] // batch_size):
                    x = vals_train[i*batch_size:(i+1)*batch_size+1,:]
                    y = train_labels[i]
                    pos = coords_train

                    edge_index , edge_attr = dense_to_sparse(adj_mat_train[i,:,:].squeeze())

                    graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos))

            elif type == "Test":
                path_test_vals = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/test_vals.mat' % n_supPix),
                path_coords_test = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/test_patch_coords.mat' % n_supPix),
                path_test_labels = os.path.join(path_main, 'datasets/MNIST_preproc_test_labels/MNIST_labels.mat'),
                path_test_centroids = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/test_centroids.mat' % n_supPix)

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



def load_Superpix75Mat(opt): #, path='../data/SuperMNIST/MNIST/datasets/mnist_superpixels_data_75/'):

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
                root='../data/SuperPix75'+ str(opt['train_size']) +type+'/', processed_file_name='GraphSuperPix75'+ str(opt['train_size'])+type+'.pt')
    type = "Test"
    Graph_test = create_Superpix75(opt, type,
                root='../data/SuperPix75'+ str(opt['test_size']) +type+'/', processed_file_name='GraphSuperPix75'+ str(opt['test_size'])+type+'.pt')

    return Graph_GNN, Graph_train, Graph_test


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_image_defaults', default='MNIST',
                        help='#Image version# Whether to run with best params for cora. Overrides the choice of dataset')
    # parser.add_argument('--use_image_defaults', action='store_true',
    #                     help='Whether to run with best params for cora. Overrides the choice of dataset')
    # parser.add_argument('--dataset', type=str, default='Cora',
    #                     help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')  ######## NEED
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
    # ODE args
    parser.add_argument('--method', type=str, default='dopri5',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")  ######## NEED
    parser.add_argument('--ode', type=str, default='ode',
                        help="set ode block. Either 'ode', 'att', 'sde'")  ######## NEED
    parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument('--simple', type=bool, default=False,
                        help='If try get rid of alpha param and the beta*x0 source term')
    # SDE args
    parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
    parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
    parser.add_argument('--adaptive', type=bool, default=False, help='use adaptive step sizes')
    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', type=bool, default=False,
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--linear_attention', type=bool, default=False,
                        help='learn the adjacency using attention at the start of each epoch, but do not update inside the ode')
    parser.add_argument('--mixed_block', type=bool, default=False,
                        help='learn the adjacency using a mix of attention and the Laplacian at the start of each epoch, but do not update inside the ode')

    # visualisation args
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--batched', type=bool, default=True,
                        help='Batching')
    parser.add_argument('--im_width', type=int, default=28, help='im_width')
    parser.add_argument('--im_height', type=int, default=28, help='im_height')
    parser.add_argument('--diags', type=bool, default=False,
                        help='Edge index include diagonal diffusion')
    parser.add_argument('--im_dataset', type=str, default='MNIST',
                        help='MNIST, CIFAR')
    args = parser.parse_args()
    opt = vars(args)
    opt = get_image_opt(opt)

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