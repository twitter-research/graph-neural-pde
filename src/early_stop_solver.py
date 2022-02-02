import torchdiffeq
from torchdiffeq._impl.dopri5 import _DORMAND_PRINCE_SHAMPINE_TABLEAU, DPS_C_MID
from torchdiffeq._impl.solvers import FixedGridODESolver
import torch
from torchdiffeq._impl.misc import _check_inputs, _flat_to_shape
import torch.nn.functional as F
import copy

from torchdiffeq._impl.interp import _interp_evaluate
from torchdiffeq._impl.rk_common import RKAdaptiveStepsizeODESolver, rk4_alt_step_func
from ogb.nodeproppred import Evaluator


def run_evaluator(evaluator, data, y_pred):
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


class EarlyStopDopri5(RKAdaptiveStepsizeODESolver):
  order = 5
  tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU
  mid = DPS_C_MID

  def __init__(self, func, y0, rtol, atol, opt, **kwargs):
    super(EarlyStopDopri5, self).__init__(func, y0, rtol, atol, **kwargs)

    self.lf = torch.nn.CrossEntropyLoss()
    self.m2_weight = None
    self.m2_bias = None
    self.data = None
    self.best_val = 0
    self.best_test = 0
    self.max_test_steps = opt['max_test_steps']
    self.best_time = 0
    self.ode_test = self.test_OGB if opt['dataset'] == 'ogbn-arxiv' else self.test
    self.dataset = opt['dataset']
    if opt['dataset'] == 'ogbn-arxiv':
      self.lf = torch.nn.functional.nll_loss
      self.evaluator = Evaluator(name=opt['dataset'])

  def set_accs(self, train, val, test, time):
    self.best_train = train
    self.best_val = val
    self.best_test = test
    self.best_time = time.item()

  def integrate(self, t):
    solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
    solution[0] = self.y0
    t = t.to(self.dtype)
    self._before_integrate(t)
    new_t = t
    for i in range(1, len(t)):
      new_t, y = self.advance(t[i])
      solution[i] = y
    return new_t, solution

  def advance(self, next_t):
    """
    Takes steps dt to get to the next user specified time point next_t. In practice this goes past next_t and then interpolates
    :param next_t:
    :return: The state, x(next_t)
    """
    n_steps = 0
    while next_t > self.rk_state.t1 and n_steps < self.max_test_steps:
      self.rk_state = self._adaptive_step(self.rk_state)
      n_steps += 1
      train_acc, val_acc, test_acc = self.evaluate(self.rk_state)
      if val_acc > self.best_val:
        self.set_accs(train_acc, val_acc, test_acc, self.rk_state.t1)
    new_t = next_t
    if n_steps < self.max_test_steps:
      return (new_t, _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t))
    else:
      return (new_t, _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, self.rk_state.t1))

  @torch.no_grad()
  def test(self, logits):
    accs = []
    for _, mask in self.data('train_mask', 'val_mask', 'test_mask'):
      pred = logits[mask].max(1)[1]
      acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
      accs.append(acc)
    return accs

  @torch.no_grad()
  def test_OGB(self, logits):
    evaluator = self.evaluator
    data = self.data
    y_pred = logits.argmax(dim=-1, keepdim=True)
    train_acc, valid_acc, test_acc = run_evaluator(evaluator, data, y_pred)
    return [train_acc, valid_acc, test_acc]

  @torch.no_grad()
  def evaluate(self, rkstate):
    # Activation.
    z = rkstate.y1
    if not self.m2_weight.shape[1] == z.shape[1]:  # system has been augmented
      z = torch.split(z, self.m2_weight.shape[1], dim=1)[0]
    z = F.relu(z)
    z = F.linear(z, self.m2_weight, self.m2_bias)
    t0, t1 = float(self.rk_state.t0), float(self.rk_state.t1)
    if self.dataset == 'ogbn-arxiv':
      z = z.log_softmax(dim=-1)
      loss = self.lf(z[self.data.train_mask], self.data.y.squeeze()[self.data.train_mask])
    else:
      loss = self.lf(z[self.data.train_mask], self.data.y[self.data.train_mask])
    train_acc, val_acc, test_acc = self.ode_test(z)
    log = 'ODE eval t0 {:.3f}, t1 {:.3f} Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(t0, t1, loss, train_acc, val_acc, tmp_test_acc))
    return train_acc, val_acc, test_acc

  def set_m2(self, m2):
    self.m2 = copy.deepcopy(m2)

  def set_data(self, data):
    if self.data is None:
      self.data = data

class EarlyStopRK4(FixedGridODESolver):
  order = 4

  def __init__(self, func, y0, opt, eps=0, **kwargs):
    super(EarlyStopRK4, self).__init__(func, y0, **kwargs)
    self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)
    self.lf = torch.nn.CrossEntropyLoss()
    self.m2_weight = None
    self.m2_bias = None
    self.data = None
    self.best_val = 0
    self.best_test = 0
    self.best_time = 0
    self.ode_test = self.test_OGB if opt['dataset'] == 'ogbn-arxiv' else self.test
    self.dataset = opt['dataset']
    if opt['dataset'] == 'ogbn-arxiv':
      self.lf = torch.nn.functional.nll_loss
      self.evaluator = Evaluator(name=opt['dataset'])

  def _step_func(self, func, t, dt, t1, y):
    ver = torchdiffeq.__version__[0] + torchdiffeq.__version__[2] + torchdiffeq.__version__[4]
    if int(ver) >= 22:  # '0.2.2'
      return rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, t1, y)
    else:
      return rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, y)

  def set_accs(self, train, val, test, time):
    self.best_train = train
    self.best_val = val
    self.best_test = test
    self.best_time = time.item()

  def integrate(self, t):
    time_grid = self.grid_constructor(self.func, self.y0, t)
    assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

    solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
    solution[0] = self.y0

    j = 1
    y0 = self.y0
    for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
      dy = self._step_func(self.func, t0, t1 - t0, t1, y0)
      y1 = y0 + dy
      train_acc, val_acc, test_acc = self.evaluate(y1, t0, t1)
      if val_acc > self.best_val:
        self.set_accs(train_acc, val_acc, test_acc, t1)

      while j < len(t) and t1 >= t[j]:
        solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
        j += 1
      y0 = y1

    return t1, solution

  @torch.no_grad()
  def test(self, logits):
    accs = []
    for _, mask in self.data('train_mask', 'val_mask', 'test_mask'):
      pred = logits[mask].max(1)[1]
      acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
      accs.append(acc)
    return accs

  @torch.no_grad()
  def test_OGB(self, logits):
    evaluator = self.evaluator
    data = self.data
    y_pred = logits.argmax(dim=-1, keepdim=True)
    train_acc, valid_acc, test_acc = run_evaluator(evaluator, data, y_pred)
    return [train_acc, valid_acc, test_acc]

  @torch.no_grad()
  def evaluate(self, z, t0, t1):
    # Activation.
    if not self.m2_weight.shape[1] == z.shape[1]:  # system has been augmented
      z = torch.split(z, self.m2_weight.shape[1], dim=1)[0]
    z = F.relu(z)
    z = F.linear(z, self.m2_weight, self.m2_bias)
    if self.dataset == 'ogbn-arxiv':
      z = z.log_softmax(dim=-1)
      loss = self.lf(z[self.data.train_mask], self.data.y.squeeze()[self.data.train_mask])
    else:
      loss = self.lf(z[self.data.train_mask], self.data.y[self.data.train_mask])
    train_acc, val_acc, test_acc = self.ode_test(z)
    log = 'ODE eval t0 {:.3f}, t1 {:.3f} Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(t0, t1, loss, train_acc, val_acc, tmp_test_acc))
    return train_acc, val_acc, test_acc

  def set_m2(self, m2):
    self.m2 = copy.deepcopy(m2)

  def set_data(self, data):
    if self.data is None:
      self.data = data


SOLVERS = {
  'dopri5': EarlyStopDopri5,
  'rk4': EarlyStopRK4
}


class EarlyStopInt(torch.nn.Module):
  def __init__(self, t, opt, device=None):
    super(EarlyStopInt, self).__init__()
    self.device = device
    self.solver = None
    self.data = None
    self.max_test_steps = opt['max_test_steps']
    self.m2_weight = None
    self.m2_bias = None
    self.opt = opt
    self.t = torch.tensor([0, opt['earlystopxT'] * t], dtype=torch.float).to(self.device)

  def __call__(self, func, y0, t, method=None, rtol=1e-7, atol=1e-9,
               adjoint_method="dopri5", adjoint_atol=1e-9, adjoint_rtol=1e-7, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a Tensor holding the state `y` and a scalar Tensor
            `t` into a Tensor of state derivatives with respect to time.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. May
            have any floating point or complex dtype.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time. May have any floating
            point dtype. Converted to a Tensor with float64 dtype.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        name: Optional name for this operation.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
        TypeError: if `options` is supplied without `method`, or if `t` or `y0` has
            an invalid dtype.
    """
    method = self.opt['method']
    assert method in ['rk4', 'dopri5'], "Only dopri5 and rk4 implemented with early stopping"

    ver = torchdiffeq.__version__
    if int(ver[0] + ver[2] + ver[4]) >= 20:  # 0.2.0 change of signature on this release for event_fn
      event_fn = None
      shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, self.t, rtol,
                                                                                                atol, method, options,
                                                                                                event_fn, SOLVERS)
    else:
      shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, self.t, rtol, atol, method, options,
                                                                     SOLVERS)

    self.solver = SOLVERS[method](func, y0, rtol=rtol, atol=atol, opt=self.opt, **options)
    if self.solver.data is None:
      self.solver.data = self.data
    self.solver.m2_weight = self.m2_weight
    self.solver.m2_bias = self.m2_bias
    t, solution = self.solver.integrate(t)
    if shapes is not None:
      solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution
