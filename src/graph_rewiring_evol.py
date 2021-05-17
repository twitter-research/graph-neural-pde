import argparse
import torch
from GNN import GNN
import time
from data import get_dataset
from ogb.nodeproppred import Evaluator

from torch_geometric.transforms.two_hop import TwoHop
from torch_geometric.transforms import GDC


def graph_evol(data):


# function to sample M nodes  - maybe use some code in AddTrainValTestMask
# __sample_split__
# function to sample K_add edges
#
# function to rank k edges
#     random
#     similarity score
#     accuracy
#
#     add randomness
#
# function to remove k_del edges
#
# population
# evaluation
# ranking/selection

    return data

@torch.no_grad()
def test(model, data, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, accs = model(feat), []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs


@torch.no_grad()
def test_OGB(model, data, opt):
  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'

  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)

  evaluator = Evaluator(name=name)
  model.eval()

  out = model(feat).log_softmax(dim=-1)
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

def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1
  # onehot[idx, labels[idx, 0]] = 1
  return torch.cat([feat, onehot], dim=-1)


def get_two_hop(data):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  th = TwoHop()
  data = th(data)
  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data


def apply_gdc(data, opt):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  print('performing gdc transformation with method {}, sparsification {}'.format(opt['gdc_method'], opt['gdc_sparsification']))
  if opt['gdc_method'] == 'ppr':
    diff_args = dict(method='ppr', alpha=opt['ppr_alpha'])
  else:
    diff_args = dict(method='heat', t=opt['heat_time'])
  if opt['gdc_sparsification'] == 'topk':
    sparse_args = dict(method='topk', k=opt['gdc_k'], dim=0)
  else:
    sparse_args = dict(method='threshold', eps=opt['gdc_threshold'])
    diff_args['eps'] = opt['gdc_threshold']
  print('gdc sparse args: {}'.format(sparse_args))
  gdc = GDC(float(opt['self_loop_weight']), normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=diff_args,
            sparsification_kwargs=sparse_args, exact=opt['exact'])
  if isinstance(data.num_nodes, list):
    data.num_nodes = data.num_nodes[0]
  data = gdc(data)
  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data






