## This code has been adapted from https://github.com/cfinlay/ffjord-rnode/ 
## MIT License

import torch
import torch.nn as nn


class RegularizedODEfunc(nn.Module):
  def __init__(self, odefunc, regularization_fns):
    super(RegularizedODEfunc, self).__init__()
    self.odefunc = odefunc
    self.regularization_fns = regularization_fns

  def before_odeint(self, *args, **kwargs):
    self.odefunc.before_odeint(*args, **kwargs)

  def forward(self, t, state):

    with torch.enable_grad():
      x = state[0]
      x.requires_grad_(True)
      t.requires_grad_(True)
      dstate = self.odefunc(t, x)
      if len(state) > 1:
        dx = dstate
        reg_states = tuple(reg_fn(x, t, dx, self.odefunc) for reg_fn in self.regularization_fns)
        return (dstate,) + reg_states
      else:
        return dstate

  @property
  def _num_evals(self):
    return self.odefunc._num_evals


def total_derivative(x, t, dx, unused_context):
  del unused_context

  directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

  try:
    u = torch.full_like(dx, 1 / x.numel(), requires_grad=True)
    tmp = torch.autograd.grad((u * dx).sum(), t, create_graph=True)[0]
    partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

    total_deriv = directional_dx + partial_dt
  except RuntimeError as e:
    if 'One of the differentiated Tensors' in e.__str__():
      raise RuntimeError(
        'No partial derivative with respect to time. Use mathematically equivalent "directional_derivative" regularizer instead')

  tdv2 = total_deriv.pow(2).view(x.size(0), -1)

  return 0.5 * tdv2.mean(dim=-1)


def directional_derivative(x, t, dx, unused_context):
  del t, unused_context

  directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
  ddx2 = directional_dx.pow(2).view(x.size(0), -1)

  return 0.5 * ddx2.mean(dim=-1)


def quadratic_cost(x, t, dx, unused_context):
  del x, t, unused_context
  dx = dx.view(dx.shape[0], -1)
  return 0.5 * dx.pow(2).mean(dim=-1)


def divergence_bf(dx, x):
  sum_diag = 0.
  for i in range(x.shape[1]):
      sum_diag += torch.autograd.grad(dx[:, i].sum(), x, create_graph=True)[0].contiguous()[:, i].contiguous()
  return sum_diag.contiguous()


def jacobian_frobenius_regularization_fn(x, t, dx, context):
  del t
  return divergence_bf(dx, x)
