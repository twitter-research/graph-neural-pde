__author__ = "Stefan Weißenberger and Johannes Klicpera"
__license__ = "MIT"

from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, InMemoryDataset


class GCN(torch.nn.Module):
    def __init__(self, opt,
                 dataset: InMemoryDataset,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GCN, self).__init__()
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

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    # def forward(self, data: Data):
    #     x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x, self.edge_index, edge_weight=self.edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)