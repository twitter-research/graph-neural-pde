"""
A GNN used at test time that supports early stopping during the integrator
"""

import torch
import torch.nn.functional as F
import argparse
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import time
from data import get_dataset
from run_GNN import get_optimizer, train, test
from early_stop_solver import EarlyStopInt
from base_classes import BaseGNN
from model_configurations import set_block, set_function


class GNNEarly(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNNEarly, self).__init__(opt, dataset, device)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.f = set_function(opt)
    self.regularization_fns = ()
    self.odeblock = block(self.f, self.regularization_fns, opt, self.data, device, t=time_tensor).to(device)
    # overwrite the test integrator with this custom one
    self.odeblock.test_integrator = EarlyStopInt(self.T, device)
    self.odeblock.test_integrator.data = self.data
    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.odeblock.train_integrator = odeint

    self.set_solver_data()

  def set_solver_m2(self):
    self.odeblock.test_integrator.m2 = self.m2

  def set_solver_data(self):
    self.odeblock.test_integrator.data = self.data

  def forward(self, x):
    # Encode each node based on its feature.
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    self.odeblock.set_x0(x)
    self.set_solver_m2()

    if self.training:
      z, self.reg_states  = self.odeblock(x)
    else:
      z = self.odeblock(x)
      
    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z


def main(opt):
  dataset = get_dataset(opt['dataset'], '../data', False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model, data = GNNEarly(opt, dataset, device).to(device), dataset.data.to(device)
  print(opt)
  # todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_val_acc = test_acc = best_epoch = 0
  best_val_acc_int = best_test_acc_int = best_epoch_int = 0
  for epoch in range(1, opt['epoch']):
    start_time = time.time()
    loss = train(model, optimizer, data)
    train_acc, val_acc, tmp_test_acc = test(model, data)
    val_acc_int = model.odeblock.test_integrator.solver.best_val
    tmp_test_acc_int = model.odeblock.test_integrator.solver.best_test
    # store best stuff inside integrator forward pass
    if val_acc_int > best_val_acc_int:
      best_val_acc_int = val_acc_int
      test_acc_int = tmp_test_acc_int
      best_epoch_int = epoch
    # store best stuff at the end of integrator forward pass
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      test_acc = tmp_test_acc
      best_epoch = epoch
    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(
      log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, tmp_test_acc))
    log = 'Performance inside integrator Val: {:.4f}, Test: {:.4f}'
    print(log.format(val_acc_int, tmp_test_acc_int))
    # print(
    # log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))
  print('best in integrator val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc_int,
                                                                                                test_acc_int,
                                                                                                best_epoch_int))


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
  # ODE args
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--ode', type=str, default='ode', help="set ode block. Either 'ode', 'att', 'sde'")
  parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument('--simple', type=bool, default=True,
                      help='If try get rid of alpha param and the beta*x0 source term')
  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=2, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')

  parser.add_argument('--linear_attention', type=bool, default=False,
                      help='learn the adjacency using attention at the start of each epoch, but do not update inside the ode')

  parser.add_argument('--jacobian-norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total-deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=0.01, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=0.01, help="int_t ||(df/dx)^T f||^2")

  args = parser.parse_args()

  opt = vars(args)

  main(opt)
