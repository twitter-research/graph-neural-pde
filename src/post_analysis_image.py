import argparse
import torch
import os
from GNN_image import GNN_image
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch_geometric.utils import to_dense_adj
import pandas as pd
from data_image import load_data
from image_opt import get_image_opt

@torch.no_grad()
def plot_image_T(model, dataset, opt, modelpath, height=2, width=3):

  loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
  fig = plt.figure() #figsize=(width*10, height*10))
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

  # plt.savefig(f"{modelpath}_imageT.png", format="PNG")
  return fig

@torch.no_grad()
def create_animation(model, dataset, opt, height, width, frames):
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
  #visualisation of ATT for the 1st image in the batch
  im_height = model.opt['im_height']
  im_width = model.opt['im_width']
  im_chan = model.opt['im_chan']
  hwc = im_height * im_width

  edge_index = model.odeblock.odefunc.edge_index
  num_nodes = model.opt['num_nodes']
  batch_size = model.opt['batch_size']
  edge_weight = model.odeblock.odefunc.edge_weight

  dense_att = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight,
                           max_num_nodes=num_nodes*batch_size)[0,:num_nodes,:num_nodes]

  square_att = dense_att.view(num_nodes, num_nodes)

  x_np = square_att.numpy()
  x_df = pd.DataFrame(x_np)
  x_df.to_csv(f"{modelpath}_att.csv")

  fig = plt.figure()
  plt.tight_layout()
  plt.imshow(square_att, cmap='hot', interpolation='nearest')
  plt.title("Attention Heat Map {}".format(model_key))
  # plt.savefig(f"{modelpath}_AttHeat.png", format="PNG")
  return fig
  # useful code to overcome normalisation colour bar
  # https: // matplotlib.org / 3.3.3 / gallery / images_contours_and_fields / multi_image.html  # sphx-glr-gallery-images-contours-and-fields-multi-image-py

@torch.no_grad()
def plot_att_edges(model):
  pass

def main(opt):
  model_key = '20210114_211047'
  directory = f"../models/"
  for filename in os.listdir(directory):
    if filename.startswith(model_key):
      path = os.path.join(directory, filename)
      print(path)
      break
  [_, _, data_name, blck, fct] = path.split("_")

  modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
  modelpath = f"{modelfolder}/model_{model_key}"

  df = pd.read_csv(f'{directory}models.csv')
  optdf = df.loc[df['model_key'] == model_key]
  opt = optdf.to_dict('records')[0]

  print("Loading Data")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data_train, data_test = load_data(opt)
  print("Loading Data")
  data_train, data_test = load_data(opt)

  print("creating GNN model")
  loader = DataLoader(data_train, batch_size=opt['batch_size'], shuffle=True)
  for batch_idx, batch in enumerate(loader):
      break
  batch.to(device)
  edge_index_gpu = batch.edge_index
  edge_attr_gpu = batch.edge_attr
  if edge_index_gpu is not None: edge_index_gpu.to(device)
  if edge_attr_gpu is not None: edge_index_gpu.to(device)

  model = GNN_image(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
                    batch.edge_attr, device).to(device)
  model.eval()

  # 1)
  fig = plot_image_T(model, data_test, opt, modelpath, height=2, width=3)
  plt.savefig(f"{modelpath}_imageT.png", format="PNG")
  # 2)
  animation = create_animation(model, data_test, opt, height=2, width=3, frames=10)
  animation.save(f'{modelpath}_animation.gif', writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=0.5)
  # 3)
  fig = plot_att_heat(model, model_key, modelpath)
  plt.savefig(f"{modelpath}_AttHeat.png", format="PNG")


if __name__ == '__main__':
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
  parser.add_argument('--reweight_attention', type=bool, default=False, help="multiply attention scores by edge weights before softmax")
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
  parser.add_argument('--gdc_threshold', type=float, default=0.0001, help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  # visualisation args
  parser.add_argument('--testing_code', type=bool, default=False, help='run on limited size training/test sets')
  parser.add_argument('--use_image_defaults', default='MNIST',help='sets as per function get_image_opt')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
  parser.add_argument('--train_size', type=int, default=128, help='Batch size')
  parser.add_argument('--test_size', type=int, default=128, help='Batch size')
  parser.add_argument('--batched', type=bool, default=True,help='Batching')
  parser.add_argument('--im_width', type=int, default=28, help='im_width')
  parser.add_argument('--im_height', type=int, default=28, help='im_height')
  parser.add_argument('--im_chan', type=int, default=1, help='im_height')
  parser.add_argument('--num_class', type=int, default=10, help='im_height')
  parser.add_argument('--diags', type=bool, default=False,help='Edge index include diagonal diffusion')
  parser.add_argument('--im_dataset', type=str, default='MNIST',help='MNIST, CIFAR')
  parser.add_argument('--num_nodes', type=int, default=28**2, help='im_width')
  args = parser.parse_args()
  opt = vars(args)
  main(opt)