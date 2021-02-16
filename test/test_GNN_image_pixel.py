#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import torch
from torch import tensor
from torch import nn
from data import get_dataset
from function_laplacian_diffusion import LaplacianODEFunc
from GNN import GNN
from block_transformer_attention import AttODEblock
from GNN_image_pixel import GNN_image_pixel
from MNIST_SuperPix import load_SuperPixel_data, train
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from utils import get_rw_adj
from run_image_pixel import pixel_test, print_model_params, get_optimizer
from torch_geometric.utils import softmax, to_dense_adj


def get_round_sum(tens, opt, n_digits=3):
  val = torch.sum(tens, dim=int(not opt['attention_norm_idx']))
  return (val * 10 ** n_digits).round() / (10 ** n_digits)


class GNNImagePixelTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2, 1], [1, 0, 1, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float)
    self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.edge1 = tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    self.x1 = torch.ones((3, 2), dtype=torch.float)

    self.leakyrelu = nn.LeakyReLU(0.2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.opt = {'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 1, 'K': 10,
                'attention_norm_idx': 0, 'simple': True, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
                'hidden_dim': 6, 'block': 'constant', 'function': 'laplacian', 'alpha_sigmoid': True, 'augment': False,
                'adjoint': False, 'tol_scale': 1, 'time': 1, 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler',
                'im_dataset': 'MNIST', 'pixel_cat': 2, 'pixel_loss': 'binary_sigmoid', 'testing_code': True,
                'batch_size': 8, 'train_size': 8, 'test_size': 8, 'kinetic_energy': None, 'jacobian_norm2': None,
                'total_deriv': None, 'directional_penalty': None, 'step_size': 1, 'max_iters': 10000,
                'max_nfe': 10000, 'reweight_attention': False, 'mix_features': False}
    if self.opt['im_dataset'] == 'MNIST':
      # http://yann.lecun.com/exdb/mnist/
      self.opt['im_width'] = 28
      self.opt['im_height'] = 28
      self.opt['im_chan'] = 1
      self.opt['hidden_dim'] = 1
      self.opt['num_feature'] = 1
      self.opt['num_class'] = 2
    # self.dataset = get_dataset('Cora', '../data', False)
    self.device = 'cpu'

  def tearDown(self) -> None:
    pass

  def test_block(self):
    data = self.dataset.data
    self.opt['hidden_dim'] = self.dataset.num_features
    self.opt['heads'] = 1
    gnn = GNN(self.opt, self.dataset, device=self.device)
    odeblock = gnn.odeblock
    self.assertTrue(isinstance(odeblock, AttODEblock))
    self.assertTrue(isinstance(odeblock.odefunc, LaplacianODEFunc))
    gnn.train()
    out = odeblock(data.x)
    self.assertTrue(data.x.shape == out.shape)
    gnn.eval()
    out = odeblock(data.x)
    print('ode block out', out)
    self.assertTrue(data.x.shape == out.shape)
    self.opt['heads'] = 2
    try:
      gnn = GNN(self.opt, self.dataset, device=self.device)
      self.assertTrue(False)
    except AssertionError:
      pass

  def test_rw_adj(self):
    dataset = load_SuperPixel_data(self.opt)
    loader = DataLoader(dataset, batch_size=self.opt['batch_size'], shuffle=True)
    for batch in loader:
      break
    model = GNN_image_pixel(self.opt, batch.num_features, batch.num_nodes, self.opt['num_class'], batch.edge_index,
                            edge_attr=None, device=torch.device('cpu'))
    test_edge_index, test_edge_weight = get_rw_adj(batch.edge_index, edge_weight=None, norm_dim=0,
                                                   fill_value=model.opt['self_loop_weight'], num_nodes=batch.num_nodes)
    self.assertTrue(test_edge_index.shape[1] == batch.edge_index.shape[1])
    self.assertTrue(test_edge_weight.shape[0] == batch.edge_index.shape[1])
    test_edge_index, test_edge_weight = get_rw_adj(batch.edge_index, edge_weight=None, norm_dim=0,
                                                   fill_value=1, num_nodes=batch.num_nodes)
    # todo this test fails - check if the dataset is constructed with self loops already
    self.assertTrue(test_edge_index.shape[1] == batch.edge_index.shape[1] + batch.x.shape[0])

  def test_gnn(self):
    dataset = load_SuperPixel_data(self.opt)
    loader = DataLoader(dataset, batch_size=self.opt['batch_size'], shuffle=True)
    for batch in loader:
      break
    self.opt['function'] = 'transformer'

    model = GNN_image_pixel(self.opt, batch.num_features, batch.num_nodes, self.opt['num_class'], batch.edge_index,
                            edge_attr=None, device=torch.device('cpu'))
    model.odeblock.odefunc.edge_index, model.odeblock.odefunc.edge_weight = get_rw_adj(batch.edge_index,
                                                                                       edge_weight=None, norm_dim=1,
                                                                                       fill_value=model.opt[
                                                                                         'self_loop_weight'],
                                                                                       num_nodes=batch.num_nodes)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer('adam', parameters, lr=0.1, weight_decay=0)
    optimizer.zero_grad()
    lf = torch.nn.CrossEntropyLoss()
    att1 = model.odeblock.odefunc.attention_weights
    self.assertTrue(att1 is None)
    model.train()
    out = model(dataset.data.x)
    self.assertTrue(out.shape[0] == batch.x.shape[0])
    self.assertTrue(torch.all(out > 0))
    self.assertTrue(torch.all(out < 1))
    self.assertTrue(torch.equal(out.sum(dim=1), torch.ones(batch.x.shape[0])))
    loss = lf(out[batch.train_mask], batch.y.squeeze()[batch.train_mask].to(model.device))
    self.assertTrue(loss.item() > 0)
    att2 = model.odeblock.odefunc.attention_weights
    self.assertTrue(att2.shape == model.odeblock.odefunc.edge_weight.shape)
    loss.backward()
    optimizer.step()
    out = model(dataset.data.x)  # regenerate attention weights from new param values
    att3 = model.odeblock.odefunc.attention_weights
    dense_attention3 = to_dense_adj(model.odeblock.odefunc.edge_index, edge_attr=att3).squeeze()
    self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention3, self.opt, n_digits=3), 1.)))

    #todo need to test if its learning non-lin attention properly


  def test_plot_superpix(self):
    dataset = load_SuperPixel_data(self.opt)
    loader = DataLoader(dataset, batch_size=self.opt['batch_size'], shuffle=True)
    for batch in loader:
      break
    self.opt['function'] = 'transformer'

    model = GNN_image_pixel(self.opt, batch.num_features, batch.num_nodes, self.opt['num_class'], batch.edge_index,
                            edge_attr=None, device=torch.device('cpu'))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer('adam', parameters, lr=0.1, weight_decay=0)
    model.train()
    for epoch in range(20):
      loss = train(model, optimizer, dataset)
      test_acc = pixel_test(model, dataset.data, batchorTest="test", trainorTest="test")
    model.eval()
    paths, atts = model.forward_plot_SuperPix(batch.x, 1)
    pass


if __name__ == '__main__':
  pass
