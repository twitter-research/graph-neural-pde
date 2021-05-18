import argparse
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from GNN import GNN, MP
from GNN_KNN import GNN_KNN
import time
from data import get_dataset
from ogb.nodeproppred import Evaluator
from graph_rewiring import apply_KNN, apply_beltrami


def get_cora_opt(opt):
  opt['dataset'] = 'Cora'
  opt['data'] = 'Planetoid'
  opt['hidden_dim'] = 16
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555
  if opt['self_loop_weight'] > 0.0:
    opt['exact'] = True  # for GDC, need exact if selp loop weight >0
  opt['alpha'] = 0.918
  opt['time'] = 12.1
  opt['num_feature'] = 1433
  opt['num_class'] = 7
  opt['num_nodes'] = 2708
  opt['epoch'] = 50
  # opt['augment'] = True
  opt['attention_dropout'] = 0
  opt['adjoint'] = False
  # opt['ode'] = 'ode'

  # opt['block'] = 'rewire_attention'
  opt['block'] = 'attention'

  return opt


def get_computers_opt(opt):
  opt['dataset'] = 'Computers'
  opt['hidden_dim'] = 16
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'adam'
  opt['lr'] = 0.01
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555
  opt['alpha'] = 0.918
  opt['epoch'] = 400
  opt['time'] = 12.1
  opt['num_feature'] = 1433
  opt['num_class'] = 7
  opt['num_nodes'] = 2708
  opt['epoch'] = 50
  opt['attention_dropout'] = 0
  opt['ode'] = 'ode'
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


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1
  # onehot[idx, labels[idx, 0]] = 1
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
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    # todo in the DGL code from OGB, they seem to apply an additional mask to the training data - currently commented
    # mask_rate = 0.5
    # mask = torch.rand(data.train_index.shape) < mask_rate
    #
    # train_pred_idx = data.train_idx[mask]
    train_pred_idx = data.train_mask

  out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
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


def train_OGB(model, mp, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    # todo in the DGL code from OGB, they seem to apply an additional mask to the training data - currently commented
    # mask_rate = 0.5
    # mask = torch.rand(data.train_index.shape) < mask_rate
    #
    # train_pred_idx = data.train_idx[mask]
    train_pred_idx = data.train_mask

  pos_encoding = mp(pos_encoding).to(model.device)
  out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
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
  logits, accs = model(feat, pos_encoding), []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


@torch.no_grad()
def test_OGB(model, mp, data, pos_encoding, opt):
  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'

  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)

  evaluator = Evaluator(name=name)
  model.eval()
  mp.eval()
  pos_encoding = mp(pos_encoding).to(model.device)

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


def main(opt):
  try:
    if opt['use_cora_defaults']:
      opt = get_cora_opt(opt)
  except KeyError:
    pass  # not always present when called as lib
  dataset = get_dataset(opt, '../data', opt['not_lcc'])
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = f"cuda:{opt['gpu']}" if torch.cuda.is_available() else 'cpu'

  if opt['beltrami'] and opt['dataset'] == 'ogbn-arxiv':
    pos_encoding = apply_beltrami(dataset.data, opt)
    mp = MP(opt, pos_encoding.shape[1], device=torch.device('cpu'))
  elif opt['beltrami']:
    pos_encoding = apply_beltrami(dataset.data, opt).to(device)
    opt['pos_enc_dim'] = pos_encoding.shape[1]
  else:
    pos_encoding = None

  if opt['rewire_KNN']:
    model = GNN_KNN(opt, dataset, device).to(device)
  else:
    model = GNN(opt, dataset, device).to(device)

  data = dataset.data.to(device)

  # todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.
  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_val_acc = test_acc = train_acc = best_epoch = 0
  # test_fn = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test

  for epoch in range(1, opt['epoch']):
    start_time = time.time()

    if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
      ei = apply_KNN(data, pos_encoding, model, opt)
      model.odeblock.odefunc.edge_index = ei

    if opt['dataset'] == 'ogbn-arxiv':  # this is a proxy for 'external encoder'
      loss = train_OGB(model, mp, optimizer, data, pos_encoding)
      train_acc, val_acc, tmp_test_acc = test_OGB(model, mp, data, pos_encoding, opt)
    else:
      loss = train(model, optimizer, data, pos_encoding)
      train_acc, val_acc, tmp_test_acc = test(model, data, pos_encoding, opt)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      test_acc = tmp_test_acc
      best_epoch = epoch
    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(
      log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))

  return train_acc, best_val_acc, test_acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  # data args
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--label_rate', type=float, default=0.5,
                      help='% of training labels to use when --use_labels is set.')
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
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention, SDE')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
  parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                      help='Add a fully connected layer to the encoder.')
  # ODE args
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=1,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
  parser.add_argument(
    "--adjoint_method", type=str, default="adaptive_heun",
    help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                      help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--adjoint_step_size', type=float, default=1,
                      help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')
  # SDE args
  parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
  parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
  parser.add_argument('--adaptive', dest='adaptive', action='store_true', help='use adaptive step sizes')
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
  parser.add_argument("--max_nfe", type=int, default=1000,
                      help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
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

  parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
  parser.add_argument('--pos_enc_type', type=str, default="GDC", help='positional encoder (default: GDC)')
  parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')
  parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
  parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
  parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
  parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
  parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--attention_type', type=str, default="scaled_dot",
                      help="scaled_dot,cosine_sim,cosine_power,pearson,rank_pearson")
  parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
  parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

  args = parser.parse_args()

  opt = vars(args)

  main(opt)
