import argparse
import torch
import torchvision
import numpy as np
from torch_geometric.nn import GCNConv, ChebConv
from GNN_image import GNN_image
import time
from torch_geometric.data import DataLoader
from data_image import edge_index_calc, create_in_memory_dataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch_geometric.utils import to_dense_adj
import pandas as pd
import torchvision.transforms as transforms
from data_image import load_data
import openpyxl
from utils import get_rw_adj

@torch.no_grad()
def print_image_T(model, dataset, opt, modelpath, height=2, width=3):

  loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
  fig = plt.figure()#figsize=(width*10, height*10))
  for batch_idx, batch in enumerate(loader):
    out = model.forward_plot_T(batch.x)
    break

  for i in range(height*width):
    # t == 0
    plt.subplot(2*height, width, i + 1)
    plt.tight_layout()
    plt.axis('off')
    mask = batch.batch == i
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(batch.x[torch.nonzero(mask)].view(model.opt['im_height'],model.opt['im_width']), cmap='gray', interpolation='none')
    elif opt['im_dataset'] == 'CIFAR':
      A = batch.x[torch.nonzero(mask)].view(model.opt['im_height'], model.opt['im_width'], model.opt['im_chan'])
      A = A / 2 + 0.5
      plt.imshow(A)
    plt.title("t=0 Ground Truth: {}".format(batch.y[i].item()))

    #t == T
    plt.subplot(2*height, width, height*width + i + 1)
    plt.tight_layout()
    plt.axis('off')
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(out[i, :].view(model.opt['im_height'], model.opt['im_width']), cmap='gray', interpolation='none')
    elif opt['im_dataset'] == 'CIFAR':
      A = out[i, :].view(model.opt['im_height'], model.opt['im_width'], model.opt['im_chan'])
      A = A / 2 + 0.5
      plt.imshow(A)
    plt.title("t=T Ground Truth: {}".format(batch.y[i].item()))

  plt.savefig(f"{modelpath}_imageT.png", format="PNG")


@torch.no_grad()
def print_image_path(model, dataset, opt, height, width, frames):
  loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
  # build the data
  for batch_idx, batch in enumerate(loader):
    paths = model.forward_plot_path(batch.x, frames)
    break
  # draw graph initial graph
  fig = plt.figure() #figsize=(width*10, height*10))
  for i in range(height * width):
    plt.subplot(height, width, i + 1)
    plt.tight_layout()
    mask = batch.batch == i
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(paths[i,0,:].view(model.opt['im_height'],model.opt['im_width']), cmap='gray', interpolation='none')
    elif opt['im_dataset'] == 'CIFAR':
      A = paths[i,0,:].view(model.opt['im_height'], model.opt['im_width'], model.opt['im_chan'])
      A = A / 2 + 0.5
      plt.imshow(A)
    plt.title("t=0 Ground Truth: {}".format(batch.y[i].item()))
    plt.axis('off')

  # loop through data and update plot
  def update(ii):
    for i in range(height * width):
      plt.subplot(height, width, i + 1)
      plt.tight_layout()
      if opt['im_dataset'] == 'MNIST':
        plt.imshow(paths[i,ii,:].view(model.opt['im_height'], model.opt['im_width']), cmap='gray', interpolation='none')
      elif opt['im_dataset'] == 'CIFAR':
        A = paths[i, ii, :].view(model.opt['im_height'], model.opt['im_width'], model.opt['im_chan'])
        A = A / 2 + 0.5   # unnormalize
        plt.imshow(A)
      plt.title("t={} Ground Truth: {}".format(ii, batch.y[i].item()))
      plt.axis('off')

  fig = plt.gcf()
  animation = FuncAnimation(fig, func=update, frames=frames, interval=10)#, blit=True)
  return animation

@torch.no_grad()
def plot_att_heat(model, model_key, modelpath):
  im_height = model.opt['im_height']
  im_width = model.opt['im_width']
  im_chan = model.opt['im_chan']
  hwc = im_height * im_width * im_chan
  slice = torch.tensor(range(hwc+1))
  edge_index = model.odeblock.edge_index
  num_nodes = model.opt['num_nodes']
  edge_weight = model.odeblock.odefunc.adj[0,:,:]

  dense_att = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)
  square_att = dense_att.view(model.opt['num_nodes'], model.opt['num_nodes'])

  x_np = square_att.numpy()
  x_df = pd.DataFrame(x_np)
  x_df.to_csv(f"{modelpath}_att.csv")

  fig = plt.figure()
  plt.tight_layout()
  plt.imshow(square_att, cmap='hot', interpolation='nearest')
  plt.title("Attention Heat Map {}".format(model_key))
  plt.savefig(f"{modelpath}_AttHeat.png", format="PNG")

  # useful code to overcome normalisation colour bar
  # https: // matplotlib.org / 3.3.3 / gallery / images_contours_and_fields / multi_image.html  # sphx-glr-gallery-images-contours-and-fields-multi-image-py

@torch.no_grad()
def plot_att_edges(model):
  pass

def main(opt):
  model_key = 'model_20210113-093420'
  modelfolder = f"../models/{model_key}"
  modelpath = f"../models/{model_key}/{model_key}"

  df = pd.read_excel('../models/models.xlsx', engine='openpyxl', )
  optdf = df.loc[df['model_key'] == model_key]
  numeric = ['batch_size', 'train_size', 'test_size', 'Test Acc', 'alpha',
             'hidden_dim', 'input_dropout', 'dropout',
             'lr', 'decay', 'self_loop_weight', 'epoch', 'time',
             'tol_scale', 'ode_blocks', 'dt_min', 'dt',
             'leaky_relu_slope', 'attention_dropout', 'heads', 'attention_norm_idx', 'attention_dim',
             'im_width', 'im_height', 'num_feature', 'num_class', 'im_chan', 'num_nodes']
  df[numeric] = df[numeric].apply(pd.to_numeric)
  opt = optdf.to_dict('records')[0]

  print("Loading Data")
  use_cuda = False
  torch.manual_seed(1)
  device = torch.device("cuda" if use_cuda else "cpu")

  Graph_GNN, Graph_train, Graph_test = load_data(opt)

  print("creating GNN model")
  # model = GNN_image(opt, Graph_GNN, device).to(device)
  #todo this is so fucked, load model with GNN to get num_classes==10 and then augment adj with below
  # loader = DataLoader(Graph_train, batch_size=model.opt['batch_size'], shuffle=True)
  loader = DataLoader(Graph_train, batch_size=opt['batch_size'], shuffle=True)
  for batch_idx, batch in enumerate(loader):
    if batch_idx == 0:# only do this for 1st batch/epoch
      # model.data = batch #loader.dataset  #adding this to reset the data
      # model.odeblock.data = batch #loader.dataset.data #why do I need to do this? duplicating data from model to ODE block?
      # model.odeblock.odefunc.adj = get_rw_adj(model.data.edge_index) #to reset adj matrix
      break

  model = GNN_image(opt, batch, opt['num_class'], device).to(device)
  # model.load_state_dict(torch.load(modelpath))
  # out = model(batch.x)
  model.eval()

  # do these as functions that take model key to generate displays on demand
  # 1)
  print_image_T(model, Graph_test, opt, modelpath, height=2, width=2) #width=3)
  # 2)
  #TODO Total Pixel intensity seems to increase loads for linear ATT
  # animation = print_image_path(model, Graph_test, opt, height=2, width=3, frames=10)
  animation = print_image_path(model, Graph_test, opt, height=2, width=2, frames=10)
  animation.save(f'{modelpath}_animation.gif', writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=0.5)
  # 3)
  # plot_att_heat(model, model_key, modelpath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_image_defaults', default='MNIST',
                      help='#Image version# Whether to run with best params for cora. Overrides the choice of dataset')
  # parser.add_argument('--use_image_defaults', action='store_true',
  #                     help='Whether to run with best params for cora. Overrides the choice of dataset')
  # parser.add_argument('--dataset', type=str, default='Cora',
  #                     help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.') ######## NEED
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
                      help="set the numerical solver: dopri5, euler, rk4, midpoint") ######## NEED
  parser.add_argument('--ode', type=str, default='ode', help="set ode block. Either 'ode', 'att', 'sde'") ######## NEED
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
  main(opt)