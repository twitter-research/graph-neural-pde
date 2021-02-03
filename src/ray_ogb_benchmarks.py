"""
Running OGB benchmarks using ray
"""
import argparse
from ray.tune import Analysis
import json
import numpy as np
from utils import get_sem, mean_confidence_interval
from ray_tune import train_ray_int
from ray import tune
from functools import partial
import os, time
from ray.tune import CLIReporter
import torch
from data import get_dataset
from run_GNN import get_optimizer, test_OGB, add_labels
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class GCN(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
               dropout):
    super(GCN, self).__init__()

    self.convs = torch.nn.ModuleList()
    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
    self.bns = torch.nn.ModuleList()
    self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
    for _ in range(num_layers - 2):
      self.convs.append(
        GCNConv(hidden_channels, hidden_channels, cached=True))
      self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
    self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    self.dropout = dropout

  def reset_parameters(self):
    for conv in self.convs:
      conv.reset_parameters()
    for bn in self.bns:
      bn.reset_parameters()

  def forward(self, x, adj_t):
    for i, conv in enumerate(self.convs[:-1]):
      x = conv(x, adj_t)
      x = self.bns[i](x)
      x = F.relu(x)
      x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.convs[-1](x, adj_t)
    return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
               dropout):
    super(SAGE, self).__init__()

    self.convs = torch.nn.ModuleList()
    self.convs.append(SAGEConv(in_channels, hidden_channels))
    self.bns = torch.nn.ModuleList()
    self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
    for _ in range(num_layers - 2):
      self.convs.append(SAGEConv(hidden_channels, hidden_channels))
      self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
    self.convs.append(SAGEConv(hidden_channels, out_channels))

    self.dropout = dropout

  def reset_parameters(self):
    for conv in self.convs:
      conv.reset_parameters()
    for bn in self.bns:
      bn.reset_parameters()

  def forward(self, x, adj_t):
    for i, conv in enumerate(self.convs[:-1]):
      x = conv(x, adj_t)
      x = self.bns[i](x)
      x = F.relu(x)
      x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.convs[-1](x, adj_t)
    return x.log_softmax(dim=-1)


def get_GNN(opt):
  if opt['gnn'] == 'gcn':
    return GCN
  elif opt['gnn'] == 'sage':
    return SAGE


def get_label_masks(idx, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx


def train(model, data, train_idx, optimizer, opt, num_classes, device):
  model.train()
  feat = data.x
  if opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(train_idx, opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, num_classes, device)

  optimizer.zero_grad()
  out = model(feat, data.adj_t)[train_idx]
  loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
  loss.backward()
  optimizer.step()

  return loss.item()


def train_ray(opt, checkpoint_dir=None, data_dir="../data"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  path = os.path.join(data_dir, 'ogbn-arxiv')
  dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=path,
                                   transform=T.ToSparseTensor())

  evaluator = Evaluator(name='ogbn-arxiv')
  data = dataset[0]
  data.adj_t = data.adj_t.to_symmetric()

  split_idx = dataset.get_idx_split()
  train_idx = split_idx['train'].to(device)

  GNN = get_GNN(opt)
  if opt['use_labels']:
    model = GNN(dataset.data.num_node_features + dataset.num_classes, opt['hidden_channels'], dataset.num_classes,
                opt['num_layers'],
                opt['dropout'])
  else:
    model = GNN(dataset.data.num_node_features, opt['hidden_channels'], dataset.num_classes, opt['num_layers'],
                opt['dropout'])

  model, data = model.to(device), data.to(device)
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])

  if checkpoint_dir:
    checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  best_time = best_epoch = train_acc = val_acc = test_acc = 0
  for epoch in range(1, opt["epoch"]):
    loss = train(model, data, train_idx, optimizer, opt, dataset.num_classes, device)
    # need next line as it sets the attributes in the solver

    tmp_train_acc, tmp_val_acc, tmp_test_acc = test(model, data, split_idx, evaluator, dataset.num_classes, opt, device)
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc

    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((model.state_dict(), optimizer.state_dict()), path)
    tune.report(loss=loss, val_acc=val_acc, test_acc=test_acc, train_acc=train_acc,
                best_epoch=best_epoch)


@torch.no_grad()
def test(model, data, split_idx, evaluator, num_classes, opt, device):
  model.eval()
  feat = data.x
  if opt['use_labels']:
    feat = add_labels(feat, data.y, split_idx['train'], num_classes, device)
  out = model(feat, data.adj_t)
  y_pred = out.argmax(dim=-1, keepdim=True)

  train_acc = evaluator.eval({
    'y_true': data.y[split_idx['train']],
    'y_pred': y_pred[split_idx['train']],
  })['acc']
  valid_acc = evaluator.eval({
    'y_true': data.y[split_idx['valid']],
    'y_pred': y_pred[split_idx['valid']],
  })['acc']
  test_acc = evaluator.eval({
    'y_true': data.y[split_idx['test']],
    'y_pred': y_pred[split_idx['test']],
  })['acc']

  return train_acc, valid_acc, test_acc


def main(opt):
  print("Running with parameters {}".format(opt))

  data_dir = os.path.abspath("../data")
  reporter = CLIReporter(
    metric_columns=["val_acc", "loss", "test_acc", "train_acc", "best_epoch", "training_iteration"])

  if opt['name'] is None:
    name = opt['folder'] + '_test'
  else:
    name = opt['name']

  result = tune.run(
    partial(train_ray, data_dir=data_dir),
    name=name,
    resources_per_trial={"cpu": 4, "gpu": 1},
    search_alg=None,
    keep_checkpoints_num=2,
    checkpoint_score_attr='val_acc',
    config=opt,
    num_samples=8,
    scheduler=None,
    max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)

  df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)

  print(df[['val_acc', 'test_acc', 'train_acc', 'best_epoch']])

  test_accs = df['test_acc'].values
  print("test accuracy {}".format(test_accs))
  log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
  print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))

  df.to_csv('../ray_results/{}_{}.csv'.format(opt['name'], time.strftime("%Y%m%d-%H%M%S")))


if __name__ == '__main__':
  opt = dict(log_steps=1, num_layers=3, optimizer='adam', hidden_channels=256, dropout=0.5, lr=0.01, epoch=500, decay=0,
             dataset='ogbn-arxiv', GDE=False, rewiring=False, use_labels=True, label_rate=0.5, metric='val_acc')
  for model in ['gcn', 'sage']:
    opt['name'] = "{}_test".format(model)
    opt['gnn'] = model
    main(opt)
