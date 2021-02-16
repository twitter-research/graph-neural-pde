import argparse
import os
import sys
import os.path as osp
import time
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils.convert import to_networkx
import skimage
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage import segmentation
from image_opt_pixel import get_image_opt, opt_perms
from GNN_image_pixel import GNN_image_pixel
from run_image_pixel import pixel_test, print_model_params, get_optimizer
from utils import get_rw_adj


def view_orig_image(SuperPixItem):
    x = SuperPixItem.orig_image
    target = SuperPixItem.target
    fig = plt.figure()
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.tight_layout()
        plt.imshow(x, cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(target.item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def view_SuperPix(SuperPixItem, opt):
    im_height = opt['im_height']
    im_width = opt['im_width']
    # RESIZING
    SF = 448 #56 #480  # 480    #needed as mark_boundaries marks 1 pixel wide either side of boundary
    heightSF = SF / im_height
    widthSF = SF / im_width
    pixel_values = SuperPixItem.orig_image.detach().numpy()
    pixel_labels = SuperPixItem.pixel_labels.view(im_height, im_width)
    x = SuperPixItem.x
    centroids = SuperPixItem.centroids
    num_centroids = torch.max(pixel_labels) + 1
    r_pixel_values, r_pixel_labels, r_centroids = transform_objects(im_height, im_width, heightSF, widthSF,
                                                                    pixel_values, pixel_labels.numpy(), centroids)
    r_y_coords, r_x_coords = get_centroid_coords_array(num_centroids, r_centroids)

    # SEGMENTATION
    r_pixel_labels = r_pixel_labels.astype(np.int)
    out = segmentation.mark_boundaries(r_pixel_values, r_pixel_labels, (1, 0, 0))

    fig, ax = plt.subplots()
    # fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.05]})
    ax.axis('off') #, cbar_ax.axis('off')
    ax.imshow(out)
    for i in range(num_centroids):
        ax.annotate(i, (r_x_coords[i], r_y_coords[i]), c="red")
    ax.scatter(x=r_x_coords, y=r_y_coords)

    NXgraph = to_networkx(SuperPixItem)
    nx.draw(NXgraph, r_centroids, ax=ax, node_size=300 / 4, edge_color="lime",
            node_color=x, cmap=plt.get_cmap('Spectral'))
    # fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('Spectral')),
    #              cax=cbar_ax, orientation="vertical")
    plt.show()


def view_dataset(opt):
    opt = get_image_opt(opt)
    SuperPixelData = load_SuperPixel_data(opt)
    # item_num = 0
    for item_num in range(4):
        SuperPixItem = SuperPixelData[item_num]
        view_orig_image(SuperPixItem)
        view_SuperPix(SuperPixItem, opt)


def get_centroid_coords_array(num_centroids, centroids):
    x_coords = []
    y_coords = []
    for i in range(num_centroids):
        x_coords.append(centroids[i][1]) #x is im_width is 1st axis
        y_coords.append(centroids[i][0]) #y is im_height is 0th axis
    return x_coords, y_coords


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


class InMemSuperPixelData(InMemoryDataset):
  def __init__(self, root, name, opt, type, transform=None, pre_transform=None, pre_filter=None):
    self.name = name
    self.opt = opt
    self.type = type
    super(InMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)
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

  def calc_centroid_values(self, num_centroids, pixel_labels, pixel_values):
      """for each centroid extract its value from its original pixel values"""
      x = []
      for c in range(num_centroids):
          centroid_value = pixel_values[np.ix_(np.where(pixel_labels == c)[0], np.where(pixel_labels == c)[1])]

          #todo problem here with ix_ on torch tensor
          x.append(torch.mean(centroid_value))
          # x.append(np.amax(centroid_value, axis=(0, 1)))

      return np.array(x)

  def calc_centroids(self, pixel_labels, num_centroids):
      centroids = {}
      for i in range(num_centroids):
          indices = np.where(pixel_labels == i)
          pixel_labels[pixel_labels == i]
          x_av = np.mean(indices[1])  # x is im_width is 1st axis
          y_av = np.mean(indices[0])  # y is im_height is 0th axis
          centroids[i] = (x_av, y_av)
      return centroids

  def find_neighbours(self, labels, boundaries, im_height, im_width):
      """return a dictionary of centroid neighbours {centroid:{neighbours}}"""
      neighbour_dict = {}
      for i in range(im_height):
          for j in range(im_width):
              pix = i * im_width + j
              if boundaries[i][j] == 1:
                  neighbours = np.array([pix - im_width, pix - 1, pix + 1, pix + im_width])
                  neighbours = neighbours[neighbours > 0]
                  neighbours = neighbours[neighbours < im_height * im_width]
                  neighbours = neighbours[
                      np.logical_or(neighbours // im_width == pix // im_width, neighbours % im_width == pix % im_width)]
                  for n in neighbours:
                      if labels[pix // im_width][pix % im_width] in neighbour_dict:
                          temp_set = neighbour_dict[labels[pix // im_width][pix % im_width]]
                          temp_set.add(labels[n // im_width][n % im_width])
                          neighbour_dict[labels[pix // im_width][pix % im_width]] = temp_set
                      else:
                          neighbour_dict[labels[pix // im_width][pix % im_width]] = {
                              labels[n // im_width][n % im_width]}
      return neighbour_dict

  def create_edge_index_from_neighbours(self, neighbour_dict, opt):
      edge_index = [[], []]
      for src, neighbours in neighbour_dict.items():
          if opt['self_loop_weight'] == 0.0:
              neighbours_temp = neighbours.copy()
              neighbours_temp.remove(src)
              edge_index[0].extend([src] * len(neighbours_temp))
              edge_index[1].extend(list(neighbours_temp))
          else:
              edge_index[0].extend([src] * len(neighbours))
              edge_index[1].extend(list(neighbours))
      return torch.tensor(edge_index, dtype=int)

  def process(self):
    graph_list = []
    data = self.read_data()
    c, w, h = self.opt['im_chan'], self.opt['im_width'], self.opt['im_height']

    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for batch_idx, (data, target) in enumerate(data_loader):
      if self.opt['testing_code'] == True and batch_idx > self.opt['train_size'] - 1:
        break
      pixel_values = data.squeeze()
      multichannel = c > 1
      #need this scaling for SLIC to work
      img  = pixel_values.numpy()
      img = img - img.min()
      img = img / img.max() * 255.0
      img = img.astype(np.double)
      # pixel_labels = slic(img, n_segments=75, multichannel=multichannel)
      compactness = 200.0 #default 10 bigger makes more square
      pixel_labels = slic(img, n_segments=60, compactness=compactness, multichannel=multichannel)
      num_centroids = np.max(pixel_labels) + 1  # required to add 1 for full coverage
      centroids = self.calc_centroids(pixel_labels, num_centroids)
      centroid_coords = torch.tensor([centroids[i] for i in range(num_centroids)])
      centroid_values = torch.tensor(self.calc_centroid_values(num_centroids, pixel_labels, pixel_values=pixel_values))
      centroid_labels = torch.tensor(np.maximum(np.minimum(centroid_values * self.opt['pixel_cat'] ,self.opt['pixel_cat']*(0.9999)), 0))
      centroid_labels = torch.floor(centroid_labels).type(torch.LongTensor)
      boundaries = skimage.segmentation.find_boundaries(pixel_labels, mode='inner', background=-1).astype(np.uint8)
      neighbour_dict = self.find_neighbours(pixel_labels, boundaries, im_height=h, im_width=w)
      edge_index = self.create_edge_index_from_neighbours(neighbour_dict, self.opt)


      # set pixel masks
      full_idx = list(centroids.keys())
      rnd_state = np.random.RandomState(seed=12345)
      train_idx = []
      for label in range(centroid_labels.max().item() + 1):
        class_idx = (centroid_labels == label).nonzero()[:,0]
        num_in_class = len(class_idx)
        # train_idx.extend(rnd_state.choice(class_idx, num_in_class//2, replace=False))
        train_idx.extend(rnd_state.choice(class_idx, min(num_in_class//2,8), replace=False))

      test_idx = [i for i in full_idx if i not in train_idx]
      train_mask = torch.zeros(num_centroids, dtype=torch.bool)
      test_mask = torch.zeros(num_centroids, dtype=torch.bool)
      train_mask[train_idx] = 1
      test_mask[test_idx] = 1
      pixel_labels = torch.tensor(pixel_labels).view(-1)

      graph = Data(x=centroid_values.unsqueeze(1), edge_index=edge_index,
                   y=centroid_labels,
                   orig_image=pixel_values, pixel_labels=pixel_labels,
                   pos=centroid_coords, centroids=centroids,
                   train_mask= train_mask, test_mask=test_mask, target=target)
      graph_list.append(graph)

    torch.save(self.collate(graph_list), self.processed_paths[0])


def load_SuperPixel_data(opt):
  print("loading PyG Super Pixel Data")
  data_name = opt['im_dataset']
  root = '../data'
  if opt['self_loop_weight'] == 0.0:
      name = f"SuperPixel{data_name}{str(opt['train_size'])}_{str(opt['pixel_cat'])}Cat_NoSL"
  else:
      name = f"SuperPixel{data_name}{str(opt['train_size'])}_{str(opt['pixel_cat'])}Cat"
  root = f"{root}/{name}"
  SuperPixelData = InMemSuperPixelData(root, name, opt, type="Train", transform=None, pre_transform=None, pre_filter=None)
  return SuperPixelData

def train(model, optimizer, dataset, data_test=None):
  model.train()
  loader = DataLoader(dataset, batch_size=model.opt['batch_size'], shuffle=True)

  for batch_idx, batch in enumerate(loader):
    optimizer.zero_grad()
    start_time = time.time()
    if batch_idx > model.opt['train_size'] // model.opt['batch_size']:  # only do for train_size data points
      break
    #TODO can i do this mid training loop or does it break the backrpop??
    model.odeblock.odefunc.edge_index, model.odeblock.odefunc.edge_weight = get_rw_adj(batch.edge_index, edge_weight=None, norm_dim=1,
                                                                   fill_value=model.opt['self_loop_weight'],
                                                                   num_nodes=batch.num_nodes)

    out = model(batch.x.to(model.device))

    if model.opt['pixel_loss'] in ['binary_sigmoid', '10catlogits', '10catM2','10catkmeans']:
      lf = torch.nn.CrossEntropyLoss()
    elif model.opt['pixel_loss'] == 'MSE':
      lf = torch.nn.MSELoss()

    if model.opt['im_chan'] == 1:
      loss = lf(out[batch.train_mask], batch.y.squeeze()[batch.train_mask].to(model.device))
    if model.opt['im_chan'] == 3:
      loss = 0
      if model.opt['pixel_cat'] == 2 and model.opt['pixel_loss'] == 'binary_sigmoid':
        loss += lf(torch.stack((out[:,0],out[:,3]),dim=1)[batch.train_mask], batch.y[:,0].squeeze()[batch.train_mask].to(model.device))
        loss += lf(torch.stack((out[:,1],out[:,4]),dim=1)[batch.train_mask], batch.y[:,1].squeeze()[batch.train_mask].to(model.device))
        loss += lf(torch.stack((out[:,2],out[:,5]),dim=1)[batch.train_mask], batch.y[:,2].squeeze()[batch.train_mask].to(model.device))

      elif model.opt['pixel_cat'] == 10 and model.opt['pixel_loss'] == '10catlogits':
        for i in range(10):
          pass #old method taking pixel slices was wrong anyway
          # loss += lf(torch.stack((out[:, i], out[:, i+9]), dim=1)[batch.train_mask].repeat(3),batch.y.view(-1,1)[batch.train_mask].repeat(3))
          # A = out[batch.train_mask.repeat(3)]
          # B = batch.y.view(-1, 1).squeeze()[batch.train_mask.repeat(3)].to(model.device)
          # loss += lf(A,B)
          # loss += lf(torch.stack((out[:, i], out[:, i + 9]), dim=1)[batch.train_mask.repeat(3)],
          #      batch.y.view(-1, 1).squeeze()[batch.train_mask.repeat(3)])

      elif model.opt['pixel_cat'] > 1 and model.opt['pixel_loss'] == '10catM2':
        A = out[batch.train_mask]
        B = batch.y.view(-1, 1).squeeze()[batch.train_mask].to(model.device)
        loss += lf(A, B)

      elif model.opt['pixel_loss'] == '10catkmeans':
        z = out[batch.train_mask].unsqueeze(1)
        batch_centers = batch.label_centers
        centers_idx = batch.batch[batch.train_mask]
        each_batch_centers = []
        pixel_cat = model.opt['pixel_cat']
        for i in range(pixel_cat):
          batch_center = batch_centers[centers_idx * pixel_cat + i].unsqueeze(1)
          each_batch_centers.append(batch_center)
        all_batch_centers = torch.cat(each_batch_centers,dim=1)
        logits = torch.sum((z-all_batch_centers)**2,dim=2)
        logits = 1 / (logits ** 2 + 1e-5)
        loss += lf(logits, batch.y[batch.train_mask].squeeze())

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()

    # if model.opt['testing_code']:
    if batch_idx % 1 == 0:
      train_acc = pixel_test(model, batch, "batch","train")
      test_acc = pixel_test(model, batch, "batch","test")
      log = 'Batch Index: {}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Test: {:.4f}'
      print(log.format(batch_idx, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, test_acc))
    #   elif batch_idx % 1 == 0:
    #     print("Batch Index {}, number of function evals {} in time {}".format(batch_idx, model.fm.sum,
    #                                                                           time.time() - start_time))
    # else:
    #   # if batch_idx % (model.opt['train_size'] / model.opt['batch_size'] / 10) == 0:
    #   # # if batch_idx % 5 == 0:
    #   #   test_acc = test(model, data_test)
    #   #   log = 'Batch Index: {}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Test: {:.4f}'
    #   #   print(log.format(batch_idx, time.time() - start_time, loss, model.fm.sum, model.bm.sum, test_acc))
    #   if batch_idx % 1 == 0:
    #     print("Batch Index {}, number of function evals {} in time {}".format(batch_idx, model.fm.sum,
    #                                                                           time.time() - start_time))
    # print("Batch Index {}, number of function evals {} in time {}".format(batch_idx, model.fm.sum,
    #                                                                       time.time() - start_time))

  return loss.item()

def main(opt):
    csv_path = '../SuperPix/models.csv'
    if os.path.exists(csv_path):
        try:
            os.rename(csv_path, '../SuperPix/temp_models.csv')
            os.rename('../SuperPix/temp_models.csv', csv_path)
        except OSError:
            print(f'Error {csv_path} is still open, please close and rerun.')
            sys.exit(1)

    opt_permutations = opt_perms(opt)
    for opt_perm, opt in opt_permutations.items():
        print(f"This is run permutation {opt['im_dataset']} {opt['block']} {opt['function']}")
        opt = get_image_opt(opt)

        print("Loading Data")
        pixel_data = load_SuperPixel_data(opt)
        print("creating GNN model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loader = DataLoader(pixel_data, batch_size=opt['batch_size'], shuffle=True)

        for batch_idx, batch in enumerate(loader):
            break
        batch.to(device)
        edge_index_gpu = batch.edge_index
        edge_attr_gpu = batch.edge_attr
        if edge_index_gpu is not None: edge_index_gpu.to(device)
        if edge_attr_gpu is not None: edge_index_gpu.to(device)
        model = GNN_image_pixel(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
                                batch.edge_attr, device).to(device)

        print(opt)
        parameters = [p for p in model.parameters() if p.requires_grad]
        print_model_params(model)
        optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])

        timestr = time.strftime("%Y%m%d_%H%M%S")
        # save model - params only - no point repeatedly saving data
        data_name = opt['im_dataset']
        blck = opt['block']
        fct = opt['function']
        model_key = f"{timestr}"
        savefolder = f"../SuperPix/{timestr}_{data_name}_{blck}_{fct}"
        savepath = f"{savefolder}/model_{model_key}"
        try:
            os.mkdir(savefolder)
        except OSError:
            print("Creation of the directory %s failed" % savefolder)
        else:
            print("Successfully created the directory %s " % savefolder)

        for epoch in range(opt['epoch']):
            print("Epoch {}".format(epoch))
            start_time = time.time()
            loss = train(model, optimizer, pixel_data)
            test_acc = pixel_test(model, pixel_data.data, batchorTest="test", trainorTest="test")

            log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Test: {:.4f}'
            print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, test_acc))

            if (epoch + 1) % 8 == 0:
                torch.save(model.state_dict(), f"{savepath}_epoch{epoch}.pt")

        # save run details to csv
        opt['model_key'] = model_key
        opt['Test Acc'] = test_acc
        df = pd.DataFrame({k: [v] for k, v in opt.items()})
        cols = list(df)
        top_cols = ['model_key', 'testing_code', 'im_dataset', 'function', 'block', 'simple', 'batched',
                    'diags', 'batch_size', 'train_size', 'test_size', 'Test Acc', 'alpha']
        for head in reversed(top_cols):
            cols.insert(0, cols.pop(cols.index(head)))
        df = df.loc[:, cols]
        header = False if os.path.exists(csv_path) else True
        df.to_csv(csv_path, mode='a', header=header)

        print(f"Test acc {test_acc}")
    return test_acc

def test_data(opt):
    opt_permutations = opt_perms(opt)
    for opt_perm, opt in opt_permutations.items():
        print(f"This is run permutation {opt['im_dataset']} {opt['block']} {opt['function']}")
        opt = get_image_opt(opt)

        data_lengths = {}
        print("Loading Data")
        pixel_data = load_SuperPixel_data(opt)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loader = DataLoader(pixel_data, batch_size=1, shuffle=False)
        for batch_idx, batch in enumerate(loader):
            if len(batch.y) in data_lengths:
                data_lengths[len(batch.y)] = data_lengths[len(batch.y)] + 1
            else:
                data_lengths[len(batch.y)] = 1

        lengths_df = pd.DataFrame.from_dict(data_lengths)
        lengths_df.to_csv(f"../SuperPix/data_lengths")
        break

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

    # view_dataset(opt)
    main(opt)
    # test_data(opt)