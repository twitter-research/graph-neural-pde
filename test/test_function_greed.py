#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test GREED attention
Note: Greed only works WITHOUT self-loops
"""
import unittest
import torch
from torch import tensor
from torch import nn
import torch_sparse
from function_greed import ODEFuncGreed
from torch_geometric.utils import softmax, to_dense_adj
from data import get_dataset
from test_params import OPT
import numpy as np
from torch_scatter import scatter_add, scatter_mul
from GNN import GNN
from run_GNN import train


class Data:
  def __init__(self, edge_index, x, y=None, train_mask=None):
    self.x = x
    self.edge_index = edge_index
    self.edge_attr = None
    self.y = y
    self.train_mask = train_mask
    self.num_features = x.shape[1]
    self.num_nodes = x.shape[0]

class DummyDataset():
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes
    self.num_features = data.num_features
    self.num_nodes = data.num_nodes


class GreedTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2, 1, 1, 3, 2, 3], [2, 0, 1, 2, 3, 1, 3, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.], [6, 7]], dtype=torch.float)
    self.y = tensor([1, 1, 0, 0])
    self.train_mask = tensor([1, 1, 1, 1])
    self.W = tensor([[0.2, 0.1], [0.3, 0.2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'dataset': 'Citeseer', 'self_loop_weight': 0, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2,
           'K': 1,
           'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
           'hidden_dim': 2, 'linear_attention': True, 'augment': False, 'adjoint': False,
           'tol_scale': 1, 'time': 1, 'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler',
           'mixed_block': False, 'max_nfe': 1000, 'mix_features': False, 'attention_dim': 2, 'rewiring': None,
           'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None, 'jacobian_norm2': None,
           'total_deriv': None, 'directional_penalty': None, 'beltrami': False}
    self.opt = {**OPT, **opt}
    self.data = Data(self.edge, self.x, self.y, self.train_mask)
    self.dataset = DummyDataset(self.data, 2)
    self.greed_func = ODEFuncGreed(2, 2, self.opt, self.data, torch.device('cpu'))

  def tearDown(self) -> None:
    pass

  def test_spvm(self):
    ei = self.greed_func.edge_index
    values = torch.rand((ei.shape[1]))
    vector = torch.rand((self.greed_func.n_nodes))
    dense_mat = to_dense_adj(ei, edge_attr=values).squeeze()
    out = self.greed_func.spvm(ei, values, vector)
    dense_out = dense_mat @ vector
    self.assertTrue(torch.all(dense_out == out))

  def test_get_R1(self):
    ei = self.greed_func.edge_index
    x = torch.rand((self.greed_func.n_nodes, 2))
    tau, tau_transpose = self.greed_func.get_tau(x)
    metric = self.greed_func.get_metric(x, tau, tau_transpose)
    gamma, _ = self.greed_func.get_gamma(metric)
    tau = tau.flatten()
    tau_transpose = tau_transpose.flatten()
    tau2 = tau * tau
    tau3 = tau * tau2
    T2 = gamma * (tau - tau3)
    T3 = gamma * (tau_transpose - tau_transpose * tau2)
    Ws = self.W @ self.W.t()
    fWs = torch.matmul(x, Ws)
    R1 = self.greed_func.get_R1(x, T2, T3, fWs)
    self.assertTrue(R1.shape[0] == self.greed_func.n_nodes)
    # get dense R1
    norm = torch.diag(self.greed_func.deg_inv_sqrt)
    T2_dense = to_dense_adj(ei, edge_attr=T2).squeeze()
    T3_dense = to_dense_adj(ei, edge_attr=T3).squeeze()
    temp = torch.diag(T2_dense.sum(dim=1))
    T4_dense = norm @ (temp - T3_dense) @ norm
    temp2 = T4_dense @ fWs
    R1_dense = torch.sum(x * temp2, dim=1)
    self.assertTrue(torch.all(torch.isclose(R1_dense, R1)))

  def test_get_R2(self):
    ei = self.greed_func.edge_index
    x = torch.rand((self.greed_func.n_nodes, 2))
    tau, tau_transpose = self.greed_func.get_tau(x)
    metric = self.greed_func.get_metric(x, tau, tau_transpose)
    gamma, _ = self.greed_func.get_gamma(metric)
    tau = tau.flatten()
    tau_transpose = tau_transpose.flatten()
    tau2 = tau * tau
    tau3 = tau * tau2
    T2 = gamma * (tau - tau3)
    T3 = gamma * (tau_transpose - tau_transpose * tau2)
    Ws = self.W @ self.W.t()
    fWs = torch.matmul(x, Ws)
    nn = self.greed_func.n_nodes
    transposed_edge_index, T3_transpose = torch_sparse.transpose(self.greed_func.edge_index, T3, nn, nn)
    T5 = self.greed_func.symmetrically_normalise(T3_transpose, transposed_edge_index)
    R2 = self.greed_func.get_R2(T2, T5, transposed_edge_index, x, fWs)
    self.assertTrue(R2.shape[0] == self.greed_func.n_nodes)
    # compare with dense
    norm = torch.diag(self.greed_func.deg_inv_sqrt)
    T2_dense = to_dense_adj(ei, edge_attr=T2).squeeze()
    T3_dense = to_dense_adj(ei, edge_attr=T3).squeeze()
    T5_dense = norm @ T3_dense.t() @ norm
    term1 = torch.sum(x * (T5_dense @ fWs), dim=1)
    norm1 = torch.diag(self.greed_func.deg_inv)
    term2 = T2_dense.t() @ (norm1 @ torch.sum(x * fWs, dim=1))
    R2_dense = term2 - term1
    self.assertTrue(torch.all(torch.isclose(R2_dense, R2)))

  def test_get_deg_inv_sqrt(self):
    deg_inv_sqrt = self.greed_func.get_deg_inv_sqrt(self.data)
    self.assertTrue(deg_inv_sqrt.shape[0] == self.data.x.shape[0])
    self.assertTrue(
      torch.all(torch.isclose(deg_inv_sqrt, torch.tensor(np.sqrt([1, 1 / 2, 1 / 3, 1 / 2]), dtype=deg_inv_sqrt.dtype)))
    )

  def test_symmetrically_normalise(self):
    values = torch.rand((self.greed_func.edge_index.shape[1]))
    normed_x = self.greed_func.symmetrically_normalise(values, self.greed_func.edge_index)
    self.assertTrue(normed_x.shape == values.shape)
    self.assertTrue(torch.all(normed_x >= 0))

  def test_T1(self):
    tau, tau_transpose = self.greed_func.get_tau(self.x)
    temp = tau * tau_transpose
    dense_tau = to_dense_adj(self.greed_func.edge_index, edge_attr=tau).squeeze()
    test1 = dense_tau * dense_tau.t()
    test2 = to_dense_adj(self.greed_func.edge_index, edge_attr=temp).squeeze()
    self.assertTrue(torch.all(test1 == test2))

  def test_T5(self):
    ei = self.greed_func.edge_index
    nn = self.greed_func.n_nodes
    T3 = torch.rand((ei.shape[1]))
    transposed_edge_index, T3_transpose = torch_sparse.transpose(ei, T3, nn, nn)
    dense_T3 = to_dense_adj(ei, edge_attr=T3).squeeze()
    norm = torch.diag(self.greed_func.deg_inv_sqrt)
    test1 = norm @ dense_T3.t() @ norm
    temp = self.greed_func.symmetrically_normalise(T3_transpose, transposed_edge_index)
    test2 = to_dense_adj(transposed_edge_index, edge_attr=temp).squeeze()
    self.assertTrue(torch.all(test1 == test2))

  def test_get_tau(self):
    tau, tau_transpose = self.greed_func.get_tau(self.x)
    self.assertTrue(tau.shape == torch.Size((self.edge.shape[1], 1)))
    self.assertTrue(tau.shape == tau_transpose.shape)
    # check that tau(2,1) is  tau_transpose(1,2) and vice versa
    self.assertTrue(tau[2] == tau_transpose[3])
    self.assertTrue(tau[3] == tau_transpose[2])
    dense_tau = to_dense_adj(self.greed_func.edge_index, edge_attr=tau).squeeze()
    dense_tau_transpose = to_dense_adj(self.greed_func.edge_index, edge_attr=tau_transpose).squeeze()
    self.assertTrue(torch.all(dense_tau.t() == dense_tau_transpose))

  def test_metric(self):
    tau, tau_transpose = self.greed_func.get_tau(self.x)
    metric = self.greed_func.get_metric(self.x, tau, tau_transpose)
    self.assertTrue(list(metric.shape) == [self.edge.shape[1]])

  def test_get_gamma(self):
    tau, tau_transpose = self.greed_func.get_tau(self.x)
    metric = self.greed_func.get_metric(self.x, tau, tau_transpose)
    epsilon = 1e-3
    gamma, _ = self.greed_func.get_gamma(metric, epsilon=epsilon)
    self.assertTrue(list(gamma.shape) == [self.edge.shape[1]])
    self.assertTrue(torch.all(gamma < 0))
    dense_gamma = to_dense_adj(self.greed_func.edge_index, edge_attr=gamma).squeeze()
    self.assertTrue(torch.all(dense_gamma == dense_gamma.t()))
    row, col = self.greed_func.edge_index
    power = self.greed_func.deg_inv[row]
    metric_power = torch.pow(metric, power)
    eta1 = scatter_mul(metric_power, row)
    eta2 = torch.pow(scatter_mul(metric, row), self.greed_func.deg_inv)
    log_eta = self.greed_func.deg_inv * scatter_add(torch.log(metric), row)
    eta3 = torch.exp(log_eta)
    self.assertTrue(torch.all(eta1 == eta2))
    self.assertTrue(torch.all(torch.isclose(eta1, eta3)))

    # currently an assertion prevents this running
    # self.opt['self_loop_weight'] = 1
    # greed_func = ODEFuncGreed(2, 2, self.opt, self.data, torch.device('cpu'))
    # gamma = greed_func.get_gamma(self.x, tau, tau_transpose, epsilon=1e-6)
    # self.assertTrue(gamma == torch.zeros(self.edge.shape[1]))

  def test_get_dynamics(self):
    tau, tau_transpose = self.greed_func.get_tau(self.x)
    metric = self.greed_func.get_metric(self.x, tau, tau_transpose)
    gamma, _ = self.greed_func.get_gamma(metric)
    W = self.greed_func.W
    Ws = W.t() @ self.W
    L, R1, R2 = self.greed_func.get_dynamics(self.x, gamma, tau, tau_transpose, Ws)
    self.assertTrue(R1.shape == R2.shape)
    self.assertTrue(R1.shape[0] == self.greed_func.n_nodes)
    # the Laplacian has one value for each edge plus one value for each self edge
    self.assertTrue(L.shape[0] == self.greed_func.n_nodes + self.greed_func.edge_index.shape[1])

  def test_get_laplacian_form(self):
    n_edges = self.greed_func.edge_index.shape[1]
    n_nodes = self.greed_func.n_nodes
    A = torch.rand((n_edges))
    D = torch.rand((n_edges))
    L = self.greed_func.get_laplacian_form(A, D)
    self.assertTrue(L.shape[0] == n_edges + n_nodes)
    self.assertTrue(torch.all(L[0:n_edges] <= 0))
    self.assertTrue(torch.all(L[n_edges:] >= 0))
    # now compare with dense calculations
    dense_A = to_dense_adj(self.greed_func.edge_index, edge_attr=A).squeeze()
    dense_D = to_dense_adj(self.greed_func.edge_index, edge_attr=D).squeeze()
    degree = torch.diag(dense_D.sum(dim=1))
    L_unorm = degree - dense_A
    normaliser = torch.diag(self.greed_func.deg_inv_sqrt)
    L_dense = normaliser @ L_unorm @ normaliser
    edges = torch.cat([self.greed_func.edge_index, self.greed_func.self_loops], dim=1)
    L_dense1 = to_dense_adj(edges, edge_attr=L).squeeze()
    self.assertTrue(torch.all(L_dense == L_dense1))

  def test_scatter_add(self):
    ei = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    degree = scatter_add(x, ei[0, :], dim=-1, dim_size=3)
    self.assertTrue(torch.all(degree == torch.tensor([3, 7, 11])))

  def test_scatter_mul(self):
    ei = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    degree = scatter_mul(x, ei[0, :], dim=-1, dim_size=3)
    self.assertTrue(torch.all(degree == torch.tensor([2, 12, 30])))

  def test_greed(self):
    self.greed_func.x0 = self.x
    f = self.greed_func(1, self.x)
    self.assertTrue(f.shape == self.x.shape)

  def test_L_goes_to_laplacian(self):
    gf = self.greed_func
    lap = gf.get_laplacian_form(torch.ones(gf.edge_index.shape[1]), torch.ones(gf.edge_index.shape[1]))
    tau, tau_transpose = gf.get_tau(self.x)
    tau = torch.ones(tau.shape)
    tau_transpose = torch.ones(tau_transpose.shape)
    metric = gf.get_metric(self.x, tau, tau_transpose)
    gamma, eta = gf.get_gamma(metric)
    gamma = -torch.ones(gamma.shape)
    W = torch.eye(gf.W.shape[0])
    Ws = W @ W.t()
    L, R1, R2 = gf.get_dynamics(self.x, gamma, tau, tau_transpose, Ws)
    self.assertTrue(torch.all(torch.eq(L, -lap)))

  def test_integration(self):
    """
    running the full pipeline on sample data
    @return:
    """
    self.opt['function'] = 'greed'
    self.opt['block'] = 'constant'
    self.opt['step_size'] = 0.1
    self.opt['time'] = 10
    self.opt['method'] = 'euler'
    self.opt['use_early'] = False
    # self.opt['attention_dim'] = 5

    #added to test_params.py
    self.opt['test_no_chanel_mix'] = True
    self.opt['test_omit_metric'] = True
    self.opt['test_mu=0'] = True
    self.opt['test_tau_remove_tanh'] = True
    self.opt['test_tau_remove_tanh_reg'] = 2
    self.opt['test_tau_symmetric'] = True
    self.opt['test_tau_remove_tanh_reg'] = 5

    gnn = GNN(self.opt, self.dataset, device=self.device)
    n_epochs = 3
    parameters = [p for p in gnn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=self.opt['lr'], weight_decay=self.opt['decay'])
    for epoch in range(n_epochs):
      loss = train(gnn, optimizer, self.dataset.data)
      print(f'loss {loss} at epoch {epoch}')
