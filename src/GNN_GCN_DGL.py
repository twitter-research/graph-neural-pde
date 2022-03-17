import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv
from torch_geometric.nn import GCNConv


class PyG_GCN(nn.Module):
    def __init__(self, in_feat, out_feat, **layer_kwargs):
        super(PyG_GCN, self).__init__()
        # layer_type(in_feat, out_feat,**layer_kwargs))
        self.GCNConv = GCNConv(in_feat, out_feat)

    def forward(self, graph, features):
        # layer(graph,features)
        x = features
        edge_index = torch.cat([graph.edges()[0].unsqueeze(0), graph.edges()[1].unsqueeze(0)], dim=0)
        return self.GCNConv(x, edge_index)
        # forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None)


class GraphSequential(nn.Module):
    def __init__(self, layer_stack):
        super().__init__()
        self.layer_stack = nn.ModuleList(layer_stack)
        self.gcn_layer_types = [GraphConv, SAGEConv, GATConv, PyG_GCN]

    def forward(self, graph, features):
        for layer in self.layer_stack:
            if any([isinstance(layer, gcn_type) for gcn_type in self.gcn_layer_types]):
                features = layer(graph, features)
            else:
                features = layer(features)
        return features


class GNNMLP(nn.Module):
    def __init__(self, opt, feat_repr_dims, enable_mlp=False, enable_gcn=True, learnable_mixing=False, use_sage=False,
                 use_gat=False, gat_num_heads=1, top_is_proj=False, use_prelu=False, dropout=0.0):
        super().__init__()
        assert not (use_sage and use_gat), 'only use sage, or gat, or neither. You can not use both'
        assert enable_mlp or enable_gcn, 'You need to have at least one of enable_gcn or enable_mlp'

        self.opt = opt
        dims = list(zip(feat_repr_dims[:-1], feat_repr_dims[1:]))
        if enable_gcn and enable_mlp:
            if learnable_mixing:
                self.mixing_coeffs = torch.nn.Parameter(torch.Tensor([0.5, 0.5]))
            else:
                self.register_buffer('mixing_coeffs', torch.Tensor([0.5, 0.5]))

        if use_prelu:
            self.nonlinearity = nn.PReLU
        else:
            self.nonlinearity = nn.ReLU
        if enable_gcn:
            if use_sage:
                gcn_layer_type = SAGEConv
                gcn_kwargs = {'aggregator_type': 'mean'}
            elif use_gat:
                gcn_layer_type = GATConv
                gcn_kwargs = {'num_heads': gat_num_heads}
            else:
                gcn_layer_type = GraphConv
                # gcn_layer_type = PyG_GCN
                gcn_kwargs = {}
            self.gcn_stack = self._make_stack(dims, top_is_proj, dropout, gcn_layer_type, gcn_kwargs)

        if enable_mlp:
            self.mlp_stack = self._make_stack(dims, top_is_proj, dropout, nn.Linear, {})

        if top_is_proj:
            self.top_proj = nn.Linear(dims[-1][0], dims[-1][1])
            if enable_gcn and enable_mlp:
                self.pre_top_proj_nonlinearity = self.nonlinearity()

        self.enable_mlp = enable_mlp
        self.enable_gcn = enable_gcn
        self.top_is_proj = top_is_proj

    def _make_stack(self, dims, top_is_proj, dropout, layer_type, layer_kwargs):
        stack_dims = dims[:-1] if top_is_proj else dims
        stack = []
        for indx, (in_feat, out_feat) in enumerate(stack_dims):
            stack.append(nn.Dropout(dropout))
            stack.append(layer_type(in_feat, out_feat, **layer_kwargs))
            if indx < len(stack_dims) - 1 or top_is_proj:
                stack.append(self.nonlinearity())
        return GraphSequential(stack)

    def forward(self, graph, features):
        if self.enable_gcn and self.enable_mlp:
            features = self.mixing_coeffs[0] * self.gcn_stack(graph, features) + self.mixing_coeffs[0] * self.mlp_stack(
                graph, features)
        elif self.enable_gcn:
            features = self.gcn_stack(graph, features)
        elif self.enable_mlp:
            features = self.mlp_stack(graph, features)

        if self.top_is_proj:
            if self.enable_gcn and self.enable_mlp:
                features = self.pre_top_proj_nonlinearity(features)
            features = self.top_proj(features)

        return features



