import os
import argparse
from argparse import Namespace
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import homophily, contains_self_loops, add_self_loops, add_remaining_self_loops, is_undirected, to_undirected
from torch_scatter import scatter_add
import torch.nn.functional as F
import wandb
from ogb.nodeproppred import Evaluator
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm

import dgl
from data_utils_DGL import load_dataset,ReadMixhopDataset
from pathlib import Path, PosixPath

from GNN import GNN
from GNN_early import GNNEarly
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
from GNN_GCN import GCN
from GNN_GCN_DGL import GNNMLP
import time, datetime
from data import get_dataset, set_train_val_test_split
from graph_rewiring import apply_KNN, apply_beltrami, apply_edge_sampling, dirichlet_energy
from best_params import best_params_dict
from greed_params import greed_test_params, greed_run_params, greed_hyper_params, greed_ablation_params, tf_ablation_args, not_sweep_args

from heterophilic import get_fixed_splits

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


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1

  return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  if data.train_mask.dtype == torch.bool:
    idx = torch.where(data.train_mask)[0]
  else:
    idx = data.train_mask
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx


def train(model, optimizer, data, pos_encoding=None):

  lf = torch.nn.functional.nll_loss if model.opt['dataset'] == 'ogbn-arxiv' else torch.nn.CrossEntropyLoss()

  if model.opt['wandb_watch_grad']: # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, lf, log="all", log_freq=10)

  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  if model.opt['function'] in ['gcn_dgl','gcn_res_dgl']:
    graph = dgl.graph((data.edge_index[0],data.edge_index[1])).to(data.edge_index.device)
    out = model(graph, feat)
  elif model.opt['function'] == 'gcn2':
    out = model(data.edge_index, feat)
  else:
    out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    # lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    # lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

  # if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
  #   reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
  #   regularization_coeffs = model.regularization_coeffs
  #
  #   reg_loss = sum(
  #     reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
  #   )
  #   loss = loss + reg_loss
  #
  # model.fm.update(model.getNFE())
  # model.resetNFE()
  # torch.autograd.set_detect_anomaly(True)
  loss.backward()#retain_graph=True)
  optimizer.step()
  # model.bm.update(model.getNFE())
  # model.resetNFE()
  return loss.item()


def train_OGB(model, mp, optimizer, data, pos_encoding=None):

  lf = torch.nn.functional.nll_loss if model.opt['dataset'] == 'ogbn-arxiv' else torch.nn.CrossEntropyLoss()

  if model.opt['wandb_watch_grad']: # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, lf, log="all", log_freq=10)

  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  pos_encoding = mp(pos_encoding).to(model.device)
  out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    # lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    # lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)


  if model.opt['function'] in ['gcn_dgl','gcn_res_dgl']:
    graph = dgl.graph((data.edge_index[0],data.edge_index[1])).to(data.edge_index.device)
    logits, accs = model(graph, feat), []
  elif model.opt['function'] == 'gcn2':
    logits, accs = model(data.edge_index, feat), []
  else:
    logits, accs = model(feat, pos_encoding), []

  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)


  if opt['function'] in ['greed', 'greed_linear', 'greed_linear_homo', 'greed_linear_hetero', 'greed_non_linear']:
    # wandb tracking
    # need to calc loss again
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(logits[data.train_mask], data.y.squeeze()[data.train_mask])

    wandb_log(data, model, opt, loss, accs[0], accs[1], accs[2], model.odeblock.odefunc.epoch)
    model.odeblock.odefunc.wandb_step = 0  # resets the wandbstep counter in function after eval forward pass
  elif opt['function'] in ['gcn','gcn2','gcn_dgl']:
    model.wandb_step = 0  # resets the wandbstep counter in function after eval forward pass

  return accs

@torch.no_grad()
def wandb_log(data, model, opt, loss, train_acc, val_acc, test_acc, epoch):

  #every epoch stats for greed linear and non linear
  num_nodes = data.num_nodes
  x0 = model.encoder(data.x)
  T0_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, x0)
  edges = torch.cat([model.odeblock.odefunc.edge_index, model.odeblock.odefunc.self_loops], dim=1)
  xN = model.forward_XN(data.x)
  TN_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, xN)
  # enc_pred = model.m2(model.encoder(data.x)).max(1)[1]
  enc_pred = model.m2(x0).max(1)[1]
  # pred = model.forward(data.x).max(1)[1]
  pred = model.m2(xN).max(1)[1]
  enc_pred_homophil = homophily(edge_index=data.edge_index, y=enc_pred)
  pred_homophil = homophily(edge_index=data.edge_index, y=pred)
  label_homophil = homophily(edge_index=data.edge_index, y=data.y)

  if opt['function'] == "greed_linear_hetero":
    LpR = model.odeblock.odefunc.L_0 + model.odeblock.odefunc.R_0
    T0_dirichlet_W = dirichlet_energy(edges, LpR, num_nodes, x0)
    TN_dirichlet_W = dirichlet_energy(edges, LpR, num_nodes, xN)

    if opt['diffusion']:
      a = model.odeblock.odefunc.mean_attention_0
      a_row_max = scatter_add(a, model.odeblock.odefunc.edge_index[0], dim=0, dim_size=num_nodes).max()
      a_row_min = scatter_add(a, model.odeblock.odefunc.edge_index[0], dim=0, dim_size=num_nodes).min()
    else:
      a_row_max = 0
      a_row_min = 0

    if opt['repulsion']:
      b = model.odeblock.odefunc.mean_attention_R0
      b_row_max = scatter_add(b, model.odeblock.odefunc.edge_index[0], dim=0, dim_size=num_nodes).max()
      b_row_min = scatter_add(b, model.odeblock.odefunc.edge_index[0], dim=0, dim_size=num_nodes).min()
    else:
      b_row_max = 0
      b_row_min = 0

    if opt['alpha_style'] == 'diag':
      alpha = model.odeblock.odefunc.alpha.mean()
    elif opt['alpha_style'] == 'free':
      alpha = model.odeblock.odefunc.alpha.data
    else:
      alpha = model.odeblock.odefunc.alpha

    if opt['wandb_track_grad_flow'] and epoch in opt['wandb_epoch_list']:
      pass
      #placeholder for if we need to apply the evolution visualisations to the linear case

    wandb.log({"loss": loss,
             # "tmp_train_acc": tmp_train_acc, "tmp_val_acc": tmp_val_acc, "tmp_test_acc": tmp_test_acc,
             "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
             "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
             "T0_dirichlet": T0_dirichlet, "TN_dirichlet": TN_dirichlet,
             "T0_dirichlet_W": T0_dirichlet_W, "TN_dirichlet_W": TN_dirichlet_W,
             "enc_pred_homophil": enc_pred_homophil, "pred_homophil": pred_homophil,
             "label_homophil": label_homophil,
             "a_row_max": a_row_max, "a_row_min": a_row_min, "b_row_max": b_row_max, "b_row_min": b_row_min,
             "alpha": alpha,
             "epoch_step": epoch})

  elif opt['function'] == "greed_non_linear":

    if opt['wandb_track_grad_flow'] and epoch in opt['wandb_epoch_list']:

      # find position of current epoch in epoch list
      idx = opt['wandb_epoch_list'].index(epoch)
      # determine index % num per page
      num_rows = 4
      row = idx % num_rows
      if row == 0: # create new figs
        spectrum_fig, spectrum_ax = plt.subplots(num_rows, 3, gridspec_kw={'width_ratios': [1,1,1]}, figsize=(24, 32))
        model.odeblock.odefunc.spectrum_fig_list.append([spectrum_fig, spectrum_ax])

        acc_entropy_fig, acc_entropy_ax = plt.subplots(num_rows, 2, gridspec_kw={'width_ratios': [1,1]}, figsize=(24, 32))
        model.odeblock.odefunc.acc_entropy_fig_list.append([acc_entropy_fig, acc_entropy_ax])

        edge_evol_fig, edge_evol_ax = plt.subplots(num_rows, 2, gridspec_kw={'width_ratios': [1,1]}, figsize=(24, 32))
        model.odeblock.odefunc.edge_evol_fig_list.append([edge_evol_fig, edge_evol_ax])
      else:
        spectrum_fig, spectrum_ax = model.odeblock.odefunc.spectrum_fig_list[-1]
        acc_entropy_fig, acc_entropy_ax = model.odeblock.odefunc.acc_entropy_fig_list[-1]
        edge_evol_fig, edge_evol_ax = model.odeblock.odefunc.edge_evol_fig_list[-1]

      #forward pass through the model in eval mode to generate the data
      model.odeblock.odefunc.get_evol_stats = True
      pred = model.forward(data.x).max(1)[1]
      model.odeblock.odefunc.get_evol_stats = False

      #spectral and accuracy plots
      Omega = model.odeblock.odefunc.Omega
      L, Q = torch.linalg.eigh(Omega)  # fast version for symmetric matrices https://pytorch.org/docs/stable/generated/torch.linalg.eig.html

      # fig, ax = plt.subplots(1, 3, figsize=(24, 8))
      # mat = ax[0].matshow(Omega, interpolation='nearest')
      # ax[0].xaxis.set_tick_params(labelsize=24)
      # ax[0].yaxis.set_tick_params(labelsize=24)
      # cbar = fig.colorbar(mat, ax=ax[0], shrink=0.75)
      # cbar.ax.tick_params(labelsize=20)
      #
      # ax[1].bar(range(L.shape[0]), L)
      # ax[1].xaxis.set_tick_params(labelsize=24)
      # ax[1].yaxis.set_tick_params(labelsize=24)
      #
      # mat2 = ax[2].matshow(Q, interpolation='nearest')
      # ax[2].xaxis.set_tick_params(labelsize=24)
      # ax[2].yaxis.set_tick_params(labelsize=24)
      # cbar1 = fig.colorbar(mat2, ax=ax[2], shrink=0.75)
      # cbar1.ax.tick_params(labelsize=20)
      # fig.suptitle(f"Omega, E-values, E-vectors, epoch {epoch}", fontsize=24)
      # fig.show()

      ###1) multi grid Omega spectrum charts
      mat = spectrum_ax[row,0].matshow(Omega, interpolation='nearest')
      spectrum_ax[row,0].xaxis.set_tick_params(labelsize=16)
      spectrum_ax[row,0].yaxis.set_tick_params(labelsize=16)
      cbar = spectrum_fig.colorbar(mat, ax=spectrum_ax[row,0], shrink=0.75)
      cbar.ax.tick_params(labelsize=16)

      spectrum_ax[row,1].bar(range(L.shape[0]), L)
      spectrum_ax[row,1].set_title(f"Omega, E-values, E-vectors, epoch {epoch}", fontdict={'fontsize':24})
      spectrum_ax[row,1].xaxis.set_tick_params(labelsize=16)
      spectrum_ax[row,1].yaxis.set_tick_params(labelsize=16)

      mat2 = spectrum_ax[row,2].matshow(Q, interpolation='nearest')
      spectrum_ax[row,2].xaxis.set_tick_params(labelsize=16)
      spectrum_ax[row,2].yaxis.set_tick_params(labelsize=16)
      cbar1 = spectrum_fig.colorbar(mat2, ax=spectrum_ax[row,2], shrink=0.75)
      cbar1.ax.tick_params(labelsize=16)
      spectrum_fig.show()

      ###2) multi grid accuracy and entropy charts
      train_accs = model.odeblock.odefunc.train_accs
      val_accs = model.odeblock.odefunc.val_accs
      test_accs = model.odeblock.odefunc.test_accs
      homophils = model.odeblock.odefunc.homophils

      # fig = plt.figure()
      # plt.plot(np.arange(0.0, len(train_accs) * opt['step_size'], opt['step_size']), train_accs, label="train")
      # plt.plot(np.arange(0.0, len(val_accs) * opt['step_size'], opt['step_size']), val_accs, label="val")
      # plt.plot(np.arange(0.0, len(test_accs) * opt['step_size'], opt['step_size']), test_accs, label="test")
      # plt.plot(np.arange(0.0, len(homophils) * opt['step_size'], opt['step_size']), homophils, label="homophil")
      # plt.title(f"Accuracy evolution, epoch {epoch}")
      # plt.legend(loc="upper right")
      # fig.show()

      acc_entropy_ax[row,0].plot(np.arange(0.0, len(train_accs) * opt['step_size'], opt['step_size']), train_accs, label="train")
      acc_entropy_ax[row,0].plot(np.arange(0.0, len(val_accs) * opt['step_size'], opt['step_size']), val_accs, label="val")
      acc_entropy_ax[row,0].plot(np.arange(0.0, len(test_accs) * opt['step_size'], opt['step_size']), test_accs, label="test")
      acc_entropy_ax[row,0].plot(np.arange(0.0, len(homophils) * opt['step_size'], opt['step_size']), homophils, label="homophil")
      acc_entropy_ax[row,0].xaxis.set_tick_params(labelsize=16)
      acc_entropy_ax[row,0].yaxis.set_tick_params(labelsize=16)
      acc_entropy_ax[row,0].set_title(f"Accuracy evolution, epoch {epoch}", fontdict={'fontsize':24})
      acc_entropy_ax[row, 0].legend(loc="upper right", fontsize=24)
      # acc_entropy_fig.show()

      #entropy plots
      #https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
      #https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
      #https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection
      entropies = model.odeblock.odefunc.entropies

      # fig, ax = plt.subplots() #figsize=(8, 16))
      x = np.arange(0.0, entropies['entropy_train_mask'].shape[0] * opt['step_size'], opt['step_size'])
      ys = entropies['entropy_train_mask'].detach().numpy()
      # ax.set_xlim(np.min(x), np.max(x))
      # ax.set_ylim(np.min(ys), np.max(ys))
      acc_entropy_ax[row, 1].set_xlim(np.min(x), np.max(x))
      acc_entropy_ax[row, 1].set_ylim(np.min(ys), np.max(ys))
      acc_entropy_ax[row,1].xaxis.set_tick_params(labelsize=16)
      acc_entropy_ax[row,1].yaxis.set_tick_params(labelsize=16)

      cmap = ListedColormap(['r', 'g'])
      norm = BoundaryNorm([-1, 0.5, 2.0], cmap.N)
      for i in range(entropies['entropy_train_mask'].shape[1]):
        tf = entropies['entropy_train_mask_correct'][:,i].float().detach().numpy()
        points = np.expand_dims(np.concatenate([x.reshape(-1,1),
                 entropies['entropy_train_mask'][:, i].reshape(-1,1)], axis=1), axis=1)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm)
        lc.set_array(tf[:-1])
        # ax.add_collection(lc)
        acc_entropy_ax[row, 1].add_collection(lc)

      # ax.set_title(f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}")
      acc_entropy_ax[row, 1].set_title(f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}", fontdict={'fontsize':24})
      acc_entropy_fig.show()

      ###3) multi grid edge value plots

      fOmf = model.odeblock.odefunc.fOmf
      # fig = plt.figure()
      # plt.plot(np.arange(0.0, fOmf.shape[0] * opt['step_size'], opt['step_size']), fOmf)
      # plt.title(f"fOmf, epoch {epoch}")
      edge_evol_ax[row,0].plot(np.arange(0.0, fOmf.shape[0] * opt['step_size'], opt['step_size']), fOmf)
      edge_evol_ax[row,0].xaxis.set_tick_params(labelsize=16)
      edge_evol_ax[row,0].yaxis.set_tick_params(labelsize=16)
      edge_evol_ax[row, 0].set_title(f"fOmf, epoch {epoch}", fontdict={'fontsize':24})

      # attentions = model.odeblock.odefunc.attentions
      # fig = plt.figure()
      # plt.plot(np.arange(0.0, attentions.shape[0] * opt['step_size'], opt['step_size']), attentions)
      # plt.title(f"Activated fOmf, epoch {epoch}")
      # fig.show()

      L2dist = model.odeblock.odefunc.L2dist
      # fig = plt.figure()
      # plt.plot(np.arange(0.0, L2dist.shape[0] * opt['step_size'], opt['step_size']), L2dist)
      # plt.title(f"L2dist, epoch {epoch}")
      edge_evol_ax[row,1].plot(np.arange(0.0, L2dist.shape[0] * opt['step_size'], opt['step_size']), L2dist)
      edge_evol_ax[row,1].xaxis.set_tick_params(labelsize=16)
      edge_evol_ax[row,1].yaxis.set_tick_params(labelsize=16)
      edge_evol_ax[row,1].set_title(f"L2dist, epoch {epoch}", fontdict={'fontsize':24})
      edge_evol_fig.show()

      model.odeblock.odefunc.fOmf = None
      model.odeblock.odefunc.attentions = None
      model.odeblock.odefunc.L2dist = None
      model.odeblock.odefunc.train_accs = None
      model.odeblock.odefunc.val_accs = None
      model.odeblock.odefunc.test_accs = None
      model.odeblock.odefunc.entropies = None

      if row == num_rows - 1:
        model.odeblock.odefunc.spectrum_pdf.savefig(spectrum_fig)
        model.odeblock.odefunc.acc_entropy_pdf.savefig(acc_entropy_fig)
        model.odeblock.odefunc.edge_evol_pdf.savefig(edge_evol_fig)

      if epoch == opt['wandb_epoch_list'][-1]:
        model.odeblock.odefunc.spectrum_pdf.close()
        model.odeblock.odefunc.acc_entropy_pdf.close()
        model.odeblock.odefunc.edge_evol_pdf.close()


    print(f"epoch {epoch}, delta: {model.odeblock.odefunc.delta.detach()}, mu: {model.odeblock.odefunc.mu}, epsilon: {model.odeblock.odefunc.om_W_eps}")  # , nu: {model.odeblock.odefunc.om_W_nu}")

    wandb.log({"loss": loss,
               # "tmp_train_acc": tmp_train_acc, "tmp_val_acc": tmp_val_acc, "tmp_test_acc": tmp_test_acc,
               "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
               "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
               "T0_dirichlet": T0_dirichlet, "TN_dirichlet": TN_dirichlet,
               "enc_pred_homophil": enc_pred_homophil, "pred_homophil": pred_homophil,
               "label_homophil": label_homophil, "delta": model.odeblock.odefunc.delta.detach(),
               # "a_row_max": a_row_max, "a_row_min": a_row_min,
               "epoch_step": epoch})
  else:
    wandb.log({"loss": loss,
               # "tmp_train_acc": tmp_train_acc, "tmp_val_acc": tmp_val_acc, "tmp_test_acc": tmp_test_acc,
               "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
               "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
               "epoch_step": epoch})


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


@torch.no_grad()
def test_OGB(model, data, pos_encoding, opt):
  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'

  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)

  evaluator = Evaluator(name=name)
  model.eval()

  out = model(feat, pos_encoding).log_softmax(dim=-1)
  y_pred = out.argmax(dim=-1, keepdim=True)

  train_acc = evaluator.eval({
    'y_true': data.y[data.train_mask],
    'y_pred': y_pred[data.train_mask],
  })['acc']
  valid_acc = evaluator.eval({
    'y_true': data.y[data.val_mask],
    'y_pred': y_pred[data.val_mask],
  })['acc']
  test_acc = evaluator.eval({
    'y_true': data.y[data.test_mask],
    'y_pred': y_pred[data.test_mask],
  })['acc']

  return train_acc, valid_acc, test_acc


def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['beltrami']:
    opt['beltrami'] = True
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] != 1:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']
  if cmd_opt['attention_type'] != '':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['max_iters'] != 100:
    opt['max_iters'] = cmd_opt['max_iters']


def main(cmd_opt):
  assert cmd_opt['function'] in ['gcn','gcn2','gcn_dgl','gcn_res_dgl'], 'script for GCN type model' #todo, incorporate the GCN logic in run_GNN

  if cmd_opt['use_best_params']:
    best_opt = best_params_dict[cmd_opt['dataset']]
    opt = {**cmd_opt, **best_opt}
    merge_cmd_args(cmd_opt, opt)
  else:
    opt = cmd_opt

  if opt['wandb']:
    if opt['use_wandb_offline']:
      os.environ["WANDB_MODE"] = "offline"
    else:
      os.environ["WANDB_MODE"] = "run"
  else:
    os.environ["WANDB_MODE"] = "disabled"  # sets as NOOP, saves keep writing: if opt['wandb']:

  # wandb_num_nans = "10"
  # os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = wandb_num_nans

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt['device'] = device


  if 'wandb_run_name' in opt.keys():
    wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
               name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True)
  else:
    wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
               reinit=True, config=opt, allow_val_change=True) #required when update hidden_dim in beltrami

  # wandb.config.update(opt, allow_val_change=True) #required when update hidden_dim in beltrami
  opt = wandb.config  # access all HPs through wandb.config, so logging matches execution!

  wandb.define_metric("epoch_step") #Customize axes - https://docs.wandb.ai/guides/track/log
  if opt['wandb_track_grad_flow']:
    wandb.define_metric("grad_flow_step") #Customize axes - https://docs.wandb.ai/guides/track/log
    wandb.define_metric("gf_e*", step_metric="grad_flow_step") #grad_flow_epoch*

  dataset = get_dataset(opt, '../data', opt['not_lcc'])
  dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)  ### added self loops for chameleon
  dataset.data.edge_index = to_undirected(dataset.data.edge_index)

  # args_dict = {'hidden_feat_repr_dims': [512, 512], 'learnable_mixing': False, 'small_train_split': False,
  #              'use_sage': False,
  #              'use_prelu': False, 'use_gat': False, 'enable_mlp_branch': False, 'enable_gcn_branch': True,
  #              'top_is_proj': False,
  #              'gat_num_heads': 1, 'make_bidirectional': True, 'iterations': 2000, 'dropout': 0.6, 'lr': 0.005,
  #              # 'weight_decay': 0.0, 'job_idx': 0, 'datasets_path': PosixPath('datasets'), 'mixhop_dataset_path': '',
  #              'weight_decay': 0.0, 'job_idx': 0, 'datasets_path': PosixPath('../datasets'), 'mixhop_dataset_path': '',
  #              # 'custom_split_file': './geom_gcn_splits/chameleon_split_0.6_0.2_0.npz', 'dataset': 'chameleon',
  #              'custom_split_file': '../geom_gcn_splits/chameleon_split_0.6_0.2_0.npz', 'dataset': 'chameleon',
  #              'self_loop': False,
  #              'split_seed': 15, 'original_split': False, 'homogeneous_split': True, 'use_synthetic_dataset': False,
  #              'syn_N0': 70,
  #              'syn_C': 10, 'syn_m': 6, 'syn_N': 10000, 'syn_homophily': 0.5, 'syn_dump_path': '',
  #              'syn_respect_original_split': False}

  args_dict = {'hidden_feat_repr_dims': int(opt['time']//opt['step_size'])*[opt['hidden_dim']], 'learnable_mixing': False, 'small_train_split': False,
               'use_sage': False,
               'use_prelu': False, 'use_gat': False, 'enable_mlp_branch': False, 'enable_gcn_branch': True,
               'top_is_proj': False,
               'gat_num_heads': 1, 'make_bidirectional': True, 'iterations': opt['epoch'], 'dropout': opt['dropout'], 'lr': opt['lr'],
               # 'weight_decay': 0.0, 'job_idx': 0, 'datasets_path': PosixPath('datasets'), 'mixhop_dataset_path': '',
               'weight_decay': opt['decay'], 'job_idx': 0, 'datasets_path': PosixPath('../datasets'), 'mixhop_dataset_path': '',
               # 'custom_split_file': './geom_gcn_splits/chameleon_split_0.6_0.2_0.npz', 'dataset': 'chameleon',
               'custom_split_file': '../geom_gcn_splits/chameleon_split_0.6_0.2_0.npz', 'dataset': 'chameleon',
               'self_loop': False,
               'split_seed': 15, 'original_split': False, 'homogeneous_split': True, 'use_synthetic_dataset': False,
               'syn_N0': 70,
               'syn_C': 10, 'syn_m': 6, 'syn_N': 10000, 'syn_homophily': 0.5, 'syn_dump_path': '',
               'syn_respect_original_split': False}

  args = Namespace(**args_dict)

  # graph, num_labels, features, labels, train_mask, val_mask, test_mask = load_dataset(args, device)
  # if args.make_bidirectional:
  #   graph = graph.to(torch.device('cpu'))
  #   graph_bi = dgl.to_bidirected(graph)
  #   for k, v in graph.ndata.items():
  #     graph_bi.ndata[k] = v
  #   graph = graph_bi.to(device)
  #
  # graph_edge_index = torch.cat([graph.edges()[0].unsqueeze(0), graph.edges()[1].unsqueeze(0)], dim=0)
  # dataset.data.edge_index = graph_edge_index
  # dataset.data.x = features
  # dataset.data.y = labels
  # dataset.data.train_mask = train_mask
  # dataset.data.val_mask = val_mask
  # dataset.data.test_mask = test_mask

  if opt['beltrami']:
    pos_encoding = apply_beltrami(dataset.data, opt).to(device)
    opt['pos_enc_dim'] = pos_encoding.shape[1]
  else:
    pos_encoding = None

  this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test

  results = []
  for rep in range(opt['num_splits']):

    if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
      dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                              num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)
    data = dataset.data.to(device)
    if opt['geom_gcn_splits']:
      data = get_fixed_splits(data, opt['dataset'], rep)

    if opt['function'] in ['gcn2', 'gcn_dgl', 'gcn_res_dgl']:
      feat_repr_dims = [dataset.data.x.shape[1]] + args.hidden_feat_repr_dims + [dataset.num_classes]
      model = GNNMLP(opt, feat_repr_dims,
             enable_mlp=args.enable_mlp_branch,
             enable_gcn=args.enable_gcn_branch,
             learnable_mixing=args.learnable_mixing,
             use_sage=args.use_sage,
             use_gat=args.use_gat,
             gat_num_heads=args.gat_num_heads,
             top_is_proj=args.top_is_proj,
             use_prelu=args.use_prelu,
             dropout=args.dropout
             ).to(device)
    elif opt['function'] in ['gcn']:
      # model = GCN(opt, dataset, hidden = [64], dropout = 0.5)
      model = GCN(opt, dataset, hidden=args.hidden_feat_repr_dims, dropout=opt['dropout']).to(device)
    else:
      if opt['rewire_KNN'] or opt['fa_layer']:
        model = GNN_KNN(opt, dataset, device).to(device) if opt["no_early"] else GNNKNNEarly(opt, dataset, device).to(
          device)
      else:
        model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)


    # parameters = [p for p in model.parameters() if p.requires_grad]
    parameters = model.parameters()

    print(opt)
    print_model_params(model)
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_time = best_epoch = train_acc = val_acc = test_acc = 0

    for epoch in range(1, opt['epoch']):
      start_time = time.time()
      if opt['function'] in ['greed', 'greed_linear', 'greed_linear_homo', 'greed_linear_hetero', 'greed_non_linear']:
        model.odeblock.odefunc.epoch = epoch
      elif opt['function'] == 'gcn':
        model.epoch = epoch

      if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
        ei = apply_KNN(data, pos_encoding, model, opt)
        model.odeblock.odefunc.edge_index = ei

      loss = train(model, optimizer, data, pos_encoding)

      tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

      best_time = opt['time']
      if tmp_val_acc > val_acc:
        best_epoch = epoch
        train_acc = tmp_train_acc
        val_acc = tmp_val_acc
        test_acc = tmp_test_acc
        best_time = opt['time']

      if not opt['function'] in ['gcn', 'gcn_dgl', 'gcn2']:
        if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
          best_epoch = epoch
          val_acc = model.odeblock.test_integrator.solver.best_val
          test_acc = model.odeblock.test_integrator.solver.best_test
          train_acc = model.odeblock.test_integrator.solver.best_train
          best_time = model.odeblock.test_integrator.solver.best_time

      print(f"Epoch: {epoch}, Runtime: {time.time() - start_time:.3f}, Loss: {loss:.3f}, "
            # f"forward nfe {model.fm.sum}, backward nfe {model.bm.sum}, "
            f"tmp_train: {tmp_train_acc:.4f}, tmp_val: {tmp_val_acc:.4f}, tmp_test: {tmp_test_acc:.4f}, "
            f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}")

      if np.isnan(loss): #todo if loss is nan do we need to wandb finish and then reinit?
        wandb_run.finish()
        # wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
        #                        reinit=True, config=opt,
        #                        allow_val_change=True)  # required when update hidden_dim in beltrami
        break

      #todo check this
      # if model.odeblock.odefunc.opt['wandb_track_grad_flow'] and epoch in opt['wandb_epoch_list']:
      #   wandb.log({f"gf_e{epoch}_attentions": wandb.plot.line_series(
      #     xs=model.odeblock.odefunc.wandb_step, ys=model.odeblock.odefunc.mean_attention_0)})


    print(f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")

    if opt['num_splits'] > 1:
      results.append([test_acc, val_acc, train_acc])

  if opt['num_splits'] > 1:
    test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                     'test_acc_std': test_acc_std}
    wandb.log(wandb_results)
    print(wandb_results)

  wandb_run.finish()
  return train_acc, val_acc, test_acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  # data args
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight', type=float, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--label_rate', type=float, default=0.5,
                      help='% of training labels to use when --use_labels is set.')
  parser.add_argument('--planetoid_split', action='store_true',
                      help='use planetoid splits for Cora/Citeseer/Pubmed')
  parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                      help='use the 10 fixed splits from '
                           'https://arxiv.org/abs/2002.05287')
  parser.add_argument('--num_splits', type=int, dest='num_splits', default=1, help='the number of splits to repeat the results on')
  # GNN args
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
  parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                      help='Add a fully connected layer to the decoder.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
  parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
  parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                      help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, help='constant, mixed, attention, hard_attention')
  parser.add_argument('--function', type=str, help='laplacian, transformer, greed, GAT')
  # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
  #                     help='Add a fully connected layer to the encoder.')
  # parser.add_argument('--add_source', dest='add_source', action='store_true',
  #                     help='If try get rid of alpha param and the beta*x0 source term')

  # ODE args
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=0.1,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
  parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                      help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
  parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                      help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--adjoint_step_size', type=float, default=1,
                      help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument("--max_nfe", type=int, default=1000,
                      help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
  # parser.add_argument("--no_early", action="store_true",
  #                     help="Whether or not to use early stopping of the ODE integrator when testing.")
  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
  parser.add_argument("--max_test_steps", type=int, default=100,
                      help="Maximum number steps for the dopri5Early test integrator. "
                           "used if getting OOM errors at test time")

  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
  parser.add_argument('--attention_type', type=str, default="",
                      help="scaled_dot,cosine_sim,pearson, exp_kernel")
  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  # rewiring args
  parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
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
  parser.add_argument('--att_samp_pct', type=float, default=1,
                      help="float in [0,1). The percentage of edges to retain based on attention scores")
  parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                      help='incorporate the feature grad in attention based edge dropout')
  parser.add_argument("--exact", action="store_true",
                      help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
  parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
  parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
  parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
  parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
  parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
  parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
  parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
  parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
  parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
  # beltrami args
  parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
  parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
  parser.add_argument('--pos_enc_type', type=str, default="DW64",
                      help='positional encoder either GDC, DW64, DW128, DW256')
  parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
  parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
  parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
  parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
  parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
  parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                      help="random, ,anchored, importance, degree")
  parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
  parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--edge_sampling_space', type=str, default="attention",
                      help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
  # parser.add_argument('--symmetric_QK', action='store_true',
  #                     help='makes the attention symmetric for rewring in QK space')
  # parser.add_argument('--symmetric_attention', action='store_true',
  #                     help='makes the attention symmetric via (A+A.T)/2')#for rewring in QK space')
  # parser.add_argument('--sym_row_max', action='store_true',
  #                     help='makes every row sum less than 1 by dividing by max rum some')
  parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
  parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
  parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")
  parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

  # wandb logging and tuning
  parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
  parser.add_argument('-wandb_offline', dest='use_wandb_offline', action='store_true')  # https://docs.wandb.ai/guides/technical-faq

  parser.add_argument('--wandb_sweep', action='store_true', help="flag if sweeping") #if not it picks up params in greed_params
  parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
  parser.add_argument('--wandb_track_grad_flow', action='store_true')

  parser.add_argument('--wandb_entity', default="graph_neural_diffusion", type=str,
                      help="jrowbottomwnb, ger__man")  # not used as default set in web browser settings
  parser.add_argument('--wandb_project', default="greed", type=str)
  parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
  parser.add_argument('--wandb_run_name', default=None, type=str)
  parser.add_argument('--wandb_output_dir', default='./wandb_output',
                      help='folder to output results, images and model checkpoints')
  # parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
  #replaces the above
  parser.add_argument('--wandb_epoch_list', nargs='+',  default=[1, 2, 4, 8, 16, 32, 64, 96, 128, 254], help='list of epochs to log gradient flow, 1 based')

  #wandb setup sweep args
  parser.add_argument('--tau_reg', type=float, default=2)
  parser.add_argument('--test_mu_0', type=str, default='True') #action='store_true')
  parser.add_argument('--test_no_chanel_mix', type=str, default='True') #action='store_true')
  parser.add_argument('--test_omit_metric_L', type=str, default='True') #action='store_true')
  parser.add_argument('--test_omit_metric_R', type=str, default='True') #action='store_true')
  parser.add_argument('--test_tau_remove_tanh', type=str, default='True') #action='store_true')
  parser.add_argument('--test_tau_symmetric', type=str, default='True') #action='store_true')
  parser.add_argument('--test_tau_outside', type=str, default='True') #action='store_true')
  parser.add_argument('--test_linear_L0', type=str, default='True') #action='store_true')
  parser.add_argument('--test_R1R2_0', type=str, default='True') #action='store_true')
  parser.add_argument('--test_grand_metric', type=str, default='True') #action='store_true')
  parser.add_argument('--test_tau_ones', type=str, default='True') #action='store_true')

  # Temp changing these to be strings so can tune over
  parser.add_argument('--use_mlp', type=str, default='False') #action='store_true')
  parser.add_argument('--add_source', type=str, default='False',
                      help='If try get rid of alpha param and the beta*x0 source term')
  parser.add_argument('--no_early', type=str, default='False') #action='store_true')
  parser.add_argument('--symmetric_QK', type=str, default='False',
                      help='makes the attention symmetric for rewring in QK space')
  parser.add_argument('--symmetric_attention', type=str, default='False',
                      help='makes the attention symmetric via (A+A.T)/2')#for rewring in QK space')
  parser.add_argument('--sym_row_max', type=str, default='False',
                      help='makes every row sum less than 1 by dividing by max rum some')

  #greed args
  parser.add_argument('--use_best_params', action='store_true', help="flag to take the best BLEND params")
  parser.add_argument('--greed_momentum', action='store_true', help="flag to use momentum grad flow")
  parser.add_argument('--momentum_alpha', type=float, default=0.2, help="alpha to use in momentum grad flow")
  parser.add_argument('--dim_p_omega', type=int, default=16, help="inner dimension for Omega")
  parser.add_argument('--dim_p_w', type=int, default=16, help="inner dimension for W")
  parser.add_argument('--gamma_epsilon', type=float, default=0.01, help="epsilon value used for numerical stability in get_gamma")


  parser.add_argument('--XN_no_activation', type=str, default='False', help='whether to relu activate the terminal state')
  parser.add_argument('--m2_mlp', type=str, default='False', help='whether to use decoder mlp')
  parser.add_argument('--attention_activation', type=str, default='exponential', help='[exponential, sigmoid] activations for the GRAM matrix')
  parser.add_argument('--attention_normalisation', type=str, default='sym_row_col', help='[mat_row_max, sym_row_col, row_bottom, "best"] how to normalise')
  parser.add_argument('--T0term_normalisation', type=str, default='T0_identity', help='[T0_symmDegnorm, T0_symmDegnorm, T0_identity] normalise T0 term')
  parser.add_argument('--laplacian_norm', type=str, default='lap_noNorm', help='[lap_symmDegnorm, lap_symmRowSumnorm, lap_noNorm] how to normalise L')
  parser.add_argument('--R_T0term_normalisation', type=str, default='T0_identity', help='[T0_symmDegnorm, T0_symmDegnorm, T0_identity] normalise T0 term')
  parser.add_argument('--R_laplacian_norm', type=str, default='lap_noNorm', help='[lap_symmDegnorm, lap_symmRowSumnorm, lap_noNorm] how to normalise L')

  parser.add_argument('--alpha_style', type=str, default='', help='"sigmoid", "free", "forced", "matrix"')
  parser.add_argument('--fix_alpha', type=float, default=None, help='control balance between diffusion and repulsion')
  parser.add_argument('--diffusion', type=str, default='True', help='turns on diffusion')
  parser.add_argument('--repulsion', type=str, default='False', help='turns on repulsion')
  parser.add_argument('--drift', type=str, default='False', help='turns on drift')
  parser.add_argument('--W_type', type=str, default='identity', help='identity, diag, full')
  parser.add_argument('--R_W_type', type=str, default='identity', help='for repulsion: identity, diag, full')
  parser.add_argument('--R_depon_A', type=str, default='', help='R dependancy in A')
  # parser.add_argument('--W_beta', type=float, default=0.5, help='for cgnn Ws orthoganal update')
  parser.add_argument('--tau_residual', type=str, default='False', help='makes tau residual')

  #GCN ablation args
  parser.add_argument('--gcn_fixed', type=str, default='False', help='fixes layers in gcn')
  parser.add_argument('--gcn_enc_dec', type=str, default='False', help='uses encoder decoder with GCN')
  parser.add_argument('--gcn_non_lin', type=str, default='False', help='uses non linearity with GCN')
  parser.add_argument('--gcn_symm', type=str, default='False', help='make weight matrix in GCN symmetric')
  parser.add_argument('--gcn_bias', type=str, default='False', help='make GCN include bias')

  # parser.add_argument('--gcn_hidden_dims', type=int,  default=1, help='number of internal GCN hidden dimensions')

  args = parser.parse_args()
  opt = vars(args)

  if opt['function'] in ['greed', 'greed_scaledDP', 'greed_linear', 'greed_linear_homo', 'greed_linear_hetero', 'greed_non_linear'
                         'gcn', 'gcn2', 'gcn_dgl', 'gcn_res_dgl']:
    opt = greed_run_params(opt)  ###basic params for GREED

    if not opt['wandb_sweep']: #sweeps are run from YAML config so don't need these
      opt = not_sweep_args(opt, project_name='greed_runs', group_name='testing')
      # this includes args for running locally - specified in YAML for tunes
      #  opt = greed_hyper_params(opt)
      # opt = greed_ablation_params(opt)

  #applied to both sweeps and not sweeps
  opt = tf_ablation_args(opt)
  main(opt)

#terminal commands for sweeps
#wandb sweep ../wandb_sweep_configs/greed_sweep_grid.yaml
#./run_sweeps.sh XXX

#--dataset texas --geom_gcn_splits --num_splits 10 --epoch 2 --function greed --use_best_params --method euler --step_size 0.25
#--dataset texas --geom_gcn_splits --num_splits 10 --epoch 2 --function greed_lin_homo --beltrami --pos_enc_type GDC --method euler --step_size 0.25 --self_loop_weight 0
#--dataset Cora --epoch 100 --function greed_linear_homo --beltrami --pos_enc_type GDC --method euler --step_size 0.25 --self_loop_weight 0 --test_tau_symmetric True
#--dataset Cora --use_best_params --function greed_linear_homo

#--dataset Cora --block attention_greed --function laplacian_greed --use_best_params --symetric_QK True --method euler --step_size 0.5 --no_early True
#--dataset Cora --block attention_greed --function laplacian_greed --use_best_params --symmetric_QK True --method euler --step_size 0.5 --no_early True
#--method euler --step_size 0.5 --no_early True

#--dataset Cora --function greed_linear_homo --use_best_params --symmetric_QK True --method euler --step_size 0.5 --no_early True
#--dataset Cora --epoch 100 --function greed_linear_homo --beltrami --pos_enc_type GDC --method euler --step_size 0.25 --self_loop_weight 0 --test_tau_symmetric True

#--dataset Cora --block attention_greed --function laplacian_greed --use_best_params --symmetric_QK True --method euler --step_size 0.5 --no_early True --attention_activation sigmoid
#--dataset Cora --epoch 100 --function greed_linear_homo --method euler --step_size 0.25 --self_loop_weight 0 --test_tau_symmetric True --symmetric_QK True --symmetric_attention False --attention_activation sigmoid --attention_normalisation sym_row_col --test_tau_ones True --use_best_params --T0term_normalisation T0_identity --T1term_normalisation T1_noNorm

#--dataset Cora --use_best_params --function greed_linear_homo --attention_activation exponential --attention_normalisation none --T0term_normalisation T0_rowSum --laplacian_norm lap_symmAttM_RowSumnorm