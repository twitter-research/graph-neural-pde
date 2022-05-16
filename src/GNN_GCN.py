__author__ = "Stefan Weißenberger and Johannes Klicpera"
__license__ = "MIT"

from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
from model_configurations import set_block, set_function
from base_classes import BaseGNN


class GCN(BaseGNN):
    def __init__(self, opt,
                 dataset: InMemoryDataset,
                 hidden: List[int] = [64],
                 dropout: float = 0.5,
                 device=None):
        ###required for code homogeniety
        super(GCN, self).__init__(opt, dataset, device)
        self.f = set_function(opt)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

        self.opt = opt
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

        self.edge_index = dataset.data.edge_index
        self.edge_attr = dataset.data.edge_attr

        self.epoch = 0
        self.wandb_step = 0

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    # def forward(self, data: Data):
    #     x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    def forward(self, x, pos_encoding): #todo pos_encoding

        for i, layer in enumerate(self.layers):
            x = self.dropout(x) #added extra dropouts to match on_local_aggregation
            x = layer(x, self.edge_index, edge_weight=self.edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            # x = self.dropout(x)
        return x
        # return torch.nn.functional.log_softmax(x, dim=1)  #cross entropy loss does not require softmax

class MLP(BaseGNN):
  def __init__(self, opt, dataset, device=None):

    ###required for code homogeniety
    super(MLP, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

    self.opt = opt
    self.m1 = Linear(dataset.data.x.shape[1], opt['hidden_dim'])
    self.m2 = Linear(opt['hidden_dim'], dataset.num_classes)

    self.epoch = 0
    self.wandb_step = 0

  def forward(self, x, pos_encoding): #todo pos_encoding
    x = F.dropout(x, self.opt['dropout'], training=self.training)
    x = F.dropout(self.m1(torch.tanh(x)), self.opt['dropout'], training=self.training)
    x = F.dropout(self.m2(torch.tanh(x)), self.opt['dropout'], training=self.training)

    return x


class GAT(BaseGNN):
    def __init__(self, opt,
                 dataset: InMemoryDataset,
                 hidden: List[int] = [64],
                 dropout: float = 0.5,
                 device=None):
        ###required for code homogeniety
        super(GAT, self).__init__(opt, dataset, device)
        self.f = set_function(opt)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

        self.opt = opt
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GATConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

        self.edge_index = dataset.data.edge_index
        self.edge_attr = dataset.data.edge_attr

        self.epoch = 0
        self.wandb_step = 0

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    # def forward(self, data: Data):
    #     x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    def forward(self, x, pos_encoding): #todo pos_encoding

        for i, layer in enumerate(self.layers):
            x = self.dropout(x) #added extra dropouts to match on_local_aggregation
            x = layer(x, self.edge_index)#, edge_weight=self.edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            # x = self.dropout(x)
        return x