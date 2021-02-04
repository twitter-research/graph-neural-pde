"""
Running OGB benchmarks using ray
"""
import argparse
from ray.tune import Analysis
import json
from torch import nn
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


class ElementWiseLinear(nn.Module):
  def __init__(self, size, weight=True, bias=True, inplace=False):
    super().__init__()
    if weight:
      self.weight = nn.Parameter(torch.Tensor(size))
    else:
      self.weight = None
    if bias:
      self.bias = nn.Parameter(torch.Tensor(size))
    else:
      self.bias = None
    self.inplace = inplace

    self.reset_parameters()

  def reset_parameters(self):
    if self.weight is not None:
      nn.init.ones_(self.weight)
    if self.bias is not None:
      nn.init.zeros_(self.bias)

  def forward(self, x):
    if self.inplace:
      if self.weight is not None:
        x.mul_(self.weight)
      if self.bias is not None:
        x.add_(self.bias)
    else:
      if self.weight is not None:
        x = x * self.weight
      if self.bias is not None:
        x = x + self.bias
    return x


class GCN(nn.Module):
  def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
    super().__init__()
    self.n_layers = n_layers
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.use_linear = use_linear

    self.convs = nn.ModuleList()
    if use_linear:
      self.linear = nn.ModuleList()
    self.norms = nn.ModuleList()

    for i in range(n_layers):
      in_hidden = n_hidden if i > 0 else in_feats
      out_hidden = n_hidden if i < n_layers - 1 else n_classes
      bias = i == n_layers - 1

      self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias))
      if use_linear:
        self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
      if i < n_layers - 1:
        self.norms.append(nn.BatchNorm1d(out_hidden))

    self.input_drop = nn.Dropout(min(0.1, dropout))
    self.dropout = nn.Dropout(dropout)
    self.activation = activation

  def forward(self, graph, feat):
    h = feat
    h = self.input_drop(h)

    for i in range(self.n_layers):
      conv = self.convs[i](graph, h)

      if self.use_linear:
        linear = self.linear[i](h)
        h = conv + linear
      else:
        h = conv

      if i < self.n_layers - 1:
        h = self.norms[i](h)
        h = self.activation(h)
        h = self.dropout(h)

    return h


class GATConv(nn.Module):
  def __init__(
          self,
          in_feats,
          out_feats,
          num_heads=1,
          feat_drop=0.0,
          attn_drop=0.0,
          edge_drop=0.0,
          negative_slope=0.2,
          use_attn_dst=True,
          residual=False,
          activation=None,
          allow_zero_in_degree=False,
          use_symmetric_norm=False,
  ):
    super(GATConv, self).__init__()
    self._num_heads = num_heads
    self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
    self._out_feats = out_feats
    self._allow_zero_in_degree = allow_zero_in_degree
    self._use_symmetric_norm = use_symmetric_norm
    if isinstance(in_feats, tuple):
      self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
      self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
    else:
      self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
    self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
    if use_attn_dst:
      self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
    else:
      self.register_buffer("attn_r", None)
    self.feat_drop = nn.Dropout(feat_drop)
    self.attn_drop = nn.Dropout(attn_drop)
    self.edge_drop = edge_drop
    self.leaky_relu = nn.LeakyReLU(negative_slope)
    if residual:
      self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
    else:
      self.register_buffer("res_fc", None)
    self.reset_parameters()
    self._activation = activation

  def reset_parameters(self):
    gain = nn.init.calculate_gain("relu")
    if hasattr(self, "fc"):
      nn.init.xavier_normal_(self.fc.weight, gain=gain)
    else:
      nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
      nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
    nn.init.xavier_normal_(self.attn_l, gain=gain)
    if isinstance(self.attn_r, nn.Parameter):
      nn.init.xavier_normal_(self.attn_r, gain=gain)
    if isinstance(self.res_fc, nn.Linear):
      nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

  def set_allow_zero_in_degree(self, set_value):
    self._allow_zero_in_degree = set_value

  def forward(self, graph, feat):
    with graph.local_scope():
      if not self._allow_zero_in_degree:
        if (graph.in_degrees() == 0).any():
          assert False

      if isinstance(feat, tuple):
        h_src = self.feat_drop(feat[0])
        h_dst = self.feat_drop(feat[1])
        if not hasattr(self, "fc_src"):
          self.fc_src, self.fc_dst = self.fc, self.fc
        feat_src, feat_dst = h_src, h_dst
        feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
        feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
      else:
        h_src = self.feat_drop(feat)
        feat_src = h_src
        feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        if graph.is_block:
          h_dst = h_src[: graph.number_of_dst_nodes()]
          feat_dst = feat_src[: graph.number_of_dst_nodes()]
        else:
          h_dst = h_src
          feat_dst = feat_src

      if self._use_symmetric_norm:
        degs = graph.out_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat_src.dim() - 1)
        norm = torch.reshape(norm, shp)
        feat_src = feat_src * norm

      # NOTE: GAT paper uses "first concatenation then linear projection"
      # to compute attention scores, while ours is "first projection then
      # addition", the two approaches are mathematically equivalent:
      # We decompose the weight vector a mentioned in the paper into
      # [a_l || a_r], then
      # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
      # Our implementation is much efficient because we do not need to
      # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
      # addition could be optimized with DGL's built-in function u_add_v,
      # which further speeds up computation and saves memory footprint.
      el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
      graph.srcdata.update({"ft": feat_src, "el": el})
      # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
      if self.attn_r is not None:
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.dstdata.update({"er": er})
        graph.apply_edges(fn.u_add_v("el", "er", "e"))
      else:
        graph.apply_edges(fn.copy_u("el", "e"))
      e = self.leaky_relu(graph.edata.pop("e"))

      if self.training and self.edge_drop > 0:
        perm = torch.randperm(graph.number_of_edges(), device=e.device)
        bound = int(graph.number_of_edges() * self.edge_drop)
        eids = perm[bound:]
        graph.edata["a"] = torch.zeros_like(e)
        graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
      else:
        graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

      # message passing
      graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
      rst = graph.dstdata["ft"]

      if self._use_symmetric_norm:
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, 0.5)
        shp = norm.shape + (1,) * (feat_dst.dim() - 1)
        norm = torch.reshape(norm, shp)
        rst = rst * norm

      # residual
      if self.res_fc is not None:
        resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
        rst = rst + resval

      # activation
      if self._activation is not None:
        rst = self._activation(rst)

      return rst

class GAT(nn.Module):
  def __init__(
          self,
          in_feats,
          n_classes,
          n_hidden,
          n_layers,
          n_heads,
          activation,
          dropout=0.0,
          input_drop=0.0,
          attn_drop=0.0,
          edge_drop=0.0,
          use_attn_dst=True,
          use_symmetric_norm=False,
  ):
    super().__init__()
    self.in_feats = in_feats
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.n_layers = n_layers
    self.num_heads = n_heads

    self.convs = nn.ModuleList()
    self.norms = nn.ModuleList()

    for i in range(n_layers):
      in_hidden = n_heads * n_hidden if i > 0 else in_feats
      out_hidden = n_hidden if i < n_layers - 1 else n_classes
      num_heads = n_heads if i < n_layers - 1 else 1
      out_channels = n_heads

      self.convs.append(
        GATConv(
          in_hidden,
          out_hidden,
          num_heads=num_heads,
          attn_drop=attn_drop,
          edge_drop=edge_drop,
          use_attn_dst=use_attn_dst,
          use_symmetric_norm=use_symmetric_norm,
          residual=True,
        )
      )

      if i < n_layers - 1:
        self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

    self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

    self.input_drop = nn.Dropout(input_drop)
    self.dropout = nn.Dropout(dropout)
    self.activation = activation

  def forward(self, graph, feat):
    h = feat
    h = self.input_drop(h)

    for i in range(self.n_layers):
      conv = self.convs[i](graph, h)

      h = conv

      if i < self.n_layers - 1:
        h = h.flatten(1)
        h = self.norms[i](h)
        h = self.activation(h, inplace=True)
        h = self.dropout(h)

    h = h.mean(1)
    h = self.bias_last(h)

    return h


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
