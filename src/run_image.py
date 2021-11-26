import os
import sys
import argparse
import torch
import torchvision
import numpy as np
from torch_geometric.nn import GCNConv, ChebConv
from GNN_image import GNN_image
import time
from torch_geometric.data import DataLoader
from data_image import edge_index_calc, create_in_memory_dataset, load_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch_geometric.utils import to_dense_adj
import pandas as pd
import torchvision.transforms as transforms
from utils import get_rw_adj
from post_analysis_image import print_image_T, print_image_path, plot_att_heat, plot_att_edges


def get_image_opt(opt):

  opt['im_dataset'] = 'MNIST' #'CIFAR'  #datasets = ['MNIST','CIFAR']
  opt['testing_code'] = True #False #True #to work with smaller dataset

  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555  #### 0?
  opt['alpha'] = 0.918
  opt['time'] = 1 #12.1
  opt['augment'] = False #True   #False need to view image
  opt['attention_dropout'] = 0
  opt['adjoint'] = False

  opt['epoch'] = 2 #3 #1 #1 #3 #10#20 #400
  opt['batch_size'] = 4 #64 #64 #64  # doing batch size for mnist
  opt['train_size'] = 32 #128#10240 #512 #10240     #128#512 #4096 #2048
  opt['test_size'] = 512 #512#64         #128
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
    opt['hidden_dim'] = 3 #16    #### 1 or 3 rgb?
    opt['num_feature'] = 3  # 1433   #### 1 or 3 rgb?
    opt['num_class'] = 10  # 7  #### mnist digits

  opt['num_nodes'] = opt['im_height'] * opt['im_width'] # * opt['im_chan'] #2708  ###pixels
  opt['simple'] = True #False does not run in new code
  opt['diags'] = True
  opt['ode'] = 'ode' #'att' don't think att is implmented properly on this codebase?
  opt['linear_attention'] = True
  opt['batched'] = True
  return opt

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))

def train(epoch, model, optimizer, dataset):

  model.train()
  loader = DataLoader(dataset, batch_size=model.opt['batch_size'], shuffle=True)

  for batch_idx, batch in enumerate(loader):
    optimizer.zero_grad()
    start_time = time.time()

    if batch_idx > model.opt['train_size']//model.opt['batch_size']: # only do this for 1st batch/epoch
      break

    out = model(batch.x)

    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out, batch.y.view(-1))  #squeeze now needed

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()
    if batch_idx % 1 == 0:
      print("Batch Index {}, number of function evals {} in time {}".format(batch_idx, model.fm.sum, time.time() - start_time))

  return loss.item()


@torch.no_grad()
def test(model, dataset):
  data = dataset.data
  test_size = data.y.shape[0]
  batch_size = model.opt['batch_size']
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  total_correct = 0
  for batch_idx, batch in enumerate(loader):
    if batch_idx > model.opt['test_size']//model.opt['batch_size']: # only do this for 1st batch/epoch
      break
    model.eval()
    logits, accs = model(batch.x), []
    pred = logits.max(1)[1]
    total_correct += pred.eq(batch.y.T).sum().item()
  accs = total_correct / test_size
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


def main(opt):

  try:
    if opt['use_image_defaults']:
      opt = get_image_opt(opt) #get_cora_opt(opt)
  except KeyError:
    pass  # not always present when called as lib

  use_cuda = False  # True  put this in opt??
  torch.manual_seed(1)
  device = torch.device("cuda" if use_cuda else "cpu")

  print("Loading Data")
  Graph_GNN, Graph_train, Graph_test = load_data(opt)

  loader = DataLoader(Graph_train, batch_size=opt['batch_size'], shuffle=True)
  for batch_idx, batch in enumerate(loader):
      break
  print("creating GNN model")
  model = GNN_image(opt, batch, opt['num_class'], device).to(device)

  print(opt)
  # todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.
  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_val_acc = test_acc = best_epoch = 0

  for epoch in range(opt['epoch']): #1, opt['epoch']):
    print("Epoch {}".format(epoch))
    start_time = time.time()

    loss = train(epoch, model, optimizer, Graph_train)
    tmp_test_acc = test(model, Graph_test)
    # train_acc, val_acc, tmp_test_acc = test(model, Graph_test)

  # Greyed this out as extra complication with batchsize linked adj
  #   if val_acc > best_val_acc:
  #     best_val_acc = val_acc
  #     test_acc = tmp_test_acc
  #     best_epoch = epoch
  #   log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
  #   print(
  #     log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
  #   # print('gamma: {}'.format(model.odeblock.gamma))
  # print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Test: {:.4f}'
    print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, tmp_test_acc))

  timestr = time.strftime("%Y%m%d-%H%M%S")

  #save model - params only - no point repeatedly saving data
  # savefolder = f"../models/model_{timestr}"
  # try:
  #   os.mkdir(savefolder)
  # except OSError:
  #   print("Creation of the directory %s failed" % savefolder)
  # else:
  #   print("Successfully created the directory %s " % savefolder)
  #
  # savepath = f"../models/model_{timestr}/model_{timestr}"
  # torch.save(model.state_dict(), savepath)

  #save params of: [opt dict, alpha, beta] to XL dataframe
  # opt['model_key'] = f"model_{timestr}"
  # opt['Test Acc'] = tmp_test_acc

  # df = pd.DataFrame({k:[v] for k,v in opt.items()})
  # cols = list(df)
  # top_cols = ['model_key','im_dataset','ode','linear_attention','batched',
  #  'simple','diags','batch_size','train_size','test_size','Test Acc' ,'alpha']#,'beta']
  # for head in reversed(top_cols):
  #   cols.insert(0, cols.pop(cols.index(head)))
  # df = df.loc[:, cols]
  # print("Saving Excel")
  # # df.to_excel("models.xlsx")
  # from excel_helper import append_df_to_excel
  # append_df_to_excel(f'../models/models.xlsx', df, sheet_name='Sheet1', startrow=None,
  #                    truncate_sheet=False,header=False) #True)#False)#,**to_excel_kwargs)
  #
  # print("creating GNN model")
  # loader = DataLoader(Graph_train, batch_size=opt['batch_size'], shuffle=True)
  # for batch_idx, batch in enumerate(loader):
  #     break
  # print("creating GNN model")
  # model = GNN_image(opt, batch, opt['num_class'], device).to(device)
  #
  # model.load_state_dict(torch.load(savepath))
  # out = model(batch.x)
  # model.eval()
  # print_image_T(model, Graph_test, opt, height=2, width=3,)
  # animation = print_image_path(model, Graph_test, height=2, width=3, frames=10)
  # animation.save(f'../images/Graph{exdataset}_ani_{timestr}.gif', writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=0.5)
  # plot_att_heat(model)
  # return tmp_test_acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
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
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
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
  parser.add_argument('--use_image_defaults', default='MNIST',
                      help='sets as per function get_image_opt')

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