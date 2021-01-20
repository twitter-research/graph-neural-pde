from torchdiffeq._impl.dopri5 import _DORMAND_PRINCE_SHAMPINE_TABLEAU, DPS_C_MID
import torch
from torchdiffeq._impl.misc import _check_inputs, _flat_to_shape
import torch.nn.functional as F
import copy

from torchdiffeq._impl.interp import _interp_evaluate
from torchdiffeq._impl.rk_common import RKAdaptiveStepsizeODESolver
from ogb.nodeproppred import Evaluator


class EarlyStopRK4(RKAdaptiveStepsizeODESolver):
  order = 5
  tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU
  mid = DPS_C_MID

  def __init__(self, func, y0, rtol, atol, opt, **kwargs):
    super(EarlyStopRK4, self).__init__(func, y0, rtol, atol, **kwargs)

    self.lf = torch.nn.CrossEntropyLoss()
    self.m2 = None
    self.data = None
    self.best_val = 0
    self.best_test = 0
    if opt['dataset'] == 'ogbn-arxiv':
      self.lf = torch.nn.functional.nll_loss
      self.ode_test = self.test_OGB
      self.evaluator = Evaluator(name=opt['dataset'])

  def set_accs(self, val, test):
    self.best_val = val
    self.best_test = test

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
    while next_t > self.rk_state.t1:
      assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
      self.rk_state = self._adaptive_step(self.rk_state)
      n_steps += 1
      val_acc, test_acc = self.evaluate(self.rk_state)
      if val_acc > self.best_val:
        self.best_val = val_acc
        self.best_test = test_acc
    new_t = next_t
    return (new_t, _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t))

  @torch.no_grad()
  def ode_test(self, logits):
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

    return [train_acc, valid_acc, test_acc]

  @torch.no_grad()
  def evaluate(self, rkstate):
    # Activation.
    z = rkstate.y1
    if not self.m2.in_features == z.shape[1]:  # system has been augmented
      z = torch.split(z, self.m2.in_features, dim=1)[0]
    z = F.relu(z)
    z = self.m2(z)
    t0, t1 = float(self.rk_state.t0), float(self.rk_state.t1)
    if self.dataset == 'ogbn-arxiv':
      z = z.log_softmax(dim=-1)
    loss = self.lf(z[self.data.train_mask], self.data.y[self.data.train_mask])
    train_acc, val_acc, tmp_test_acc = self.ode_test(z)
    log = 'ODE eval t0 {:.3f}, t1 {:.3f} Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(t0, t1, loss, train_acc, val_acc, tmp_test_acc))
    return val_acc, tmp_test_acc

  def set_m2(self, m2):
    self.m2 = copy.deepcopy(m2)

  def set_data(self, data):
    if self.data is None:
      self.data = data


SOLVERS = {
  'early_stopping': EarlyStopRK4
}


class EarlyStopInt(torch.nn.Module):
  def __init__(self, t, opt, device=None):
    super(EarlyStopInt, self).__init__()
    self.device = device
    self.solver = None
    self.data = None
    self.m2 = None
    self.opt = opt
    self.t = torch.tensor([0, 3 * t], dtype=torch.float).to(self.device)

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
    method = 'early_stopping'
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, self.t, rtol, atol, method, options,
                                                                     SOLVERS)

    self.solver = SOLVERS[method](func, y0, rtol=rtol, atol=atol, opt=self.opt, **options)
    if self.solver.data is None:
      self.solver.data = self.data
    self.solver.m2 = self.m2
    t, solution = self.solver.integrate(t)
    if shapes is not None:
      solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution
