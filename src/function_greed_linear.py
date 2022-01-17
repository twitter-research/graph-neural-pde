"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""

import torch
from torch import nn
import torch_sparse
from torch_scatter import scatter_add, scatter_mul
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.utils import degree
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
import wandb
from function_greed import ODEFuncGreed
from utils import MaxNFEException
from base_classes import ODEFunc


class ODEFuncGreedLinear(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreedLinear, self).__init__(in_features, out_features, opt, data, device, bias=False)

    self.energy = 0
    self.epoch = 0
    self.wandb_step = 0
    self.prev_grad = None
    self.x_0 = None
    self.L_0 = None #static Laplacian form

  def set_x_0(self, x_0):
    self.x_0 = x_0.clone().detach()

  def set_tau_0(self, x_0):
    self.tau_0, self.tau_transpose_0 = self.get_tau(x_0)

  def set_L0(self, x_0, tau_0, tau_transpose_0):
    # tau_0, tau_transpose_0 = self.get_tau(self.x_0)
    metric_0 = self.get_metric(x_0, tau_0, tau_transpose_0)
    if self.opt['test_omit_metric']:
      self.eta_0 = torch.ones(metric_0.shape, device=x_0.device)
      self.gamma_0 = -torch.ones(metric_0.shape, device=x_0.device)  # setting metric equal to adjacency
    else:
      self.gamma_0, self.eta_0 = self.get_gamma(metric_0, self.opt['gamma_epsilon'])
    self.L_0 = self.get_laplacian_linear(self.gamma_0, tau_0, tau_transpose_0)

  def get_laplacian_linear(self, gamma, tau, tau_transpose):
    """
    These dynamics follow directly from differentiating the energy
    The structure of T0,T1 etc. is a direct result of differentiating the hyperbolic tangent nonlinearity.
    Note: changing the nonlinearity will change the equations
    Returns: L - a generalisation of the Laplacian, but using attention
             R1, R2 are a result of tau being a function of the features and both are zero if tau is not dependent on
             features
    """
    tau = tau.flatten()
    tau_transpose = tau_transpose.flatten()
    tau2 = tau * tau
    if self.opt['test_tau_remove_tanh']:
      T0 = gamma * tau2
      T1 = gamma * tau * tau_transpose
    else:
      T0 = gamma * tau2
      T1 = gamma * tau * tau_transpose
    L = self.get_laplacian_form(T1, T0)
    return L

  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    Ws = self.W @ self.W.t()  # output a [d,d] tensor

    if self.opt['test_linear_L0']:
      with torch.no_grad(): #these things go into calculating energy not forward
        metric_0 = self.get_metric(self.x_0, self.tau_0, self.tau_transpose_0)
        _, eta = self.get_gamma(metric_0, self.opt['gamma_epsilon'])
        tau, tau_transpose = self.tau_0, self.tau_transpose_0 #assigning for energy calcs
      edges = torch.cat([self.edge_index, self.self_loops], dim=1)
      L = self.L_0
      f = torch_sparse.spmm(edges, L, x.shape[0], x.shape[0], x)
    else:
      tau, tau_transpose = self.get_tau(x)
      metric = self.get_metric(self.x_0, tau, tau_transpose) #metric is dependent on x_0 but still on time implicitly through tau
      gamma, eta = self.get_gamma(metric, self.opt['gamma_epsilon'])
      edges = torch.cat([self.edge_index, self.self_loops], dim=1)
      L = self.get_laplacian_linear(gamma, self.tau_0, self.tau_transpose_0)
      f = torch_sparse.spmm(edges, L, x.shape[0], x.shape[0], x)

    R1 = 0
    R2 = 0
    f = torch.matmul(f, Ws)

    f = f - 0.5 * self.mu * (x - self.x0)
    # f = f + self.x0
    # todo consider adding a term f = f + self.alpha * f

    if self.opt['test_omit_metric'] and self.opt['test_mu_0']: #energy to use when Gamma is -adjacency and not the pullback and mu == 0
      energy = torch.sum(self.get_energy_gradient(x, tau, tau_transpose) ** 2)
    elif self.opt['test_omit_metric']: #energy to use when Gamma is -adjacency and not the pullback and mu != 0
      energy = torch.sum(self.get_energy_gradient(x, tau, tau_transpose) ** 2) + self.mu * torch.sum((x - self.x0) ** 2)
    else:
      energy = self.get_energy(x, eta)

    if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and self.training:
      wandb.log({f"gf_e{self.epoch}_energy_change": energy - self.energy, f"gf_e{self.epoch}_energy": energy,
                 f"gf_e{self.epoch}_f": f ** 2, f"gf_e{self.epoch}_L": torch.sum(L ** 2),
                 f"gf_e{self.epoch}_R1": torch.sum(R1 ** 2), f"gf_e{self.epoch}_R2": torch.sum(R2 ** 2), f"gf_e{self.epoch}_mu": self.mu,
                 "grad_flow_step": self.wandb_step}) #, step=self.wandb_step)
      # todo Customize axes - https://docs.wandb.ai/guides/track/log
      self.wandb_step += 1

    self.energy = energy

    if self.opt['greed_momentum'] and self.prev_grad:
      f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
      self.prev_grad = f
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
