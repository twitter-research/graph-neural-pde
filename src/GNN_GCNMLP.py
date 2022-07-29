import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
import dgl
# from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv
from torch_geometric.nn import GCNConv
from model_configurations import set_block, set_function
from base_classes import BaseGNN

#code adapted from https://github.com/kdd-submitter/on_local_aggregation/blob/main/models.py - converted to PyG

class PyG_GCN(nn.Module):
    def __init__(self, in_feat, out_feat, **layer_kwargs):
        super(PyG_GCN, self).__init__()
        self.GCNConv = GCNConv(in_feat, out_feat)

    def forward(self, graph, features):
        edge_index = torch.cat([graph.edges()[0].unsqueeze(0), graph.edges()[1].unsqueeze(0)], dim=0)
        return self.GCNConv(features, edge_index)


class GraphSequential(nn.Module):
    def __init__(self, layer_stack, opt):
        super().__init__()
        self.layer_stack = nn.ModuleList(layer_stack)
        self.gcn_layer_types = [GraphConv, SAGEConv, GATConv, PyG_GCN, GCNConv]#, ResGraphConv]
        self.opt = opt

    def forward(self, graph, features):
        for layer in self.layer_stack:
            if any([isinstance(layer, gcn_type) for gcn_type in self.gcn_layer_types]):
                if self.opt['gcn_symm']:
                    # encoder conv
                    if layer._in_feats != layer._out_feats:
                        features = layer(graph, features)
                    # symmetric gcn
                    elif self.opt['function'] == 'gcn_dgl':
                        symm_weight = (layer.symm_weight + layer.symm_weight.T) / 2
                        features = layer(graph, features, weight=symm_weight)
                    # symmetric res gcn
                    elif self.opt['function'] == 'gcn_res_dgl':
                        symm_weight = (layer.symm_weight + layer.symm_weight.T) / 2
                        features = features + self.opt['step_size'] * layer(graph, features, weight=symm_weight)
                else:
                    if self.opt['function'] == 'gcn_dgl':
                        features = layer(graph, features)
                    elif self.opt['function'] == 'gcn_res_dgl':
                        if layer._in_feats != layer._out_feats:
                            features = layer(graph, features)
                        else:
                            features = features + self.opt['step_size'] * layer(graph, features)
                    elif self.opt['function'] == 'gcn2':
                        features = layer(features, graph) ##here "graph" is edge index
            else:
                features = layer(features)
        return features


class GNNMLP(BaseGNN):
    def __init__(self, opt, dataset, device, feat_repr_dims, enable_mlp=False, enable_gcn=True, learnable_mixing=False, use_sage=False,
                 use_gat=False, gat_num_heads=1, top_is_proj=False, use_prelu=False, dropout=0.0):
        ###required for code homogeniety
        super(GNNMLP, self).__init__(opt, dataset, device)
        self.f = set_function(opt)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
        ###

        assert not (use_sage and use_gat), 'only use sage, or gat, or neither. You can not use both'
        assert enable_mlp or enable_gcn, 'You need to have at least one of enable_gcn or enable_mlp'

        if opt['gcn_symm']:
            assert opt['function'] in ['gcn_dgl','gcn_res_dgl'], 'gcn_symm only implenent for dgl conv'
        assert not enable_mlp, 'Not using this branch in our code'

        self.opt = opt
        self.edge_index = dataset.data.edge_index

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
                ### choice of PyG and DGL GCNConv
                if self.opt['function'] == 'gcn_dgl':
                    gcn_layer_type = GraphConv
                elif self.opt['function'] == 'gcn_res_dgl':
                    gcn_layer_type = GraphConv #ResGraphConv
                elif self.opt['function'] == 'gcn2':
                    # gcn_layer_type = PyG_GCN #changed input signature instead of using slow wrapper
                    gcn_layer_type = GCNConv

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

        if self.opt['gcn_enc_dec']:
            # stack.append(nn.Dropout(dropout))
            # stack.append(nn.Linear(stack_dims[0][0], stack_dims[0][1]))
            self.m1 = nn.Linear(stack_dims[0][0], stack_dims[0][1])

            stack_dims = dims[1:-1]

        #initialise the fixed shared layer if required
        if self.opt['gcn_fixed']:
            if self.opt['gcn_enc_dec']:
                if self.opt['gcn_symm']:
                    #init layer without weights
                    GCN_fixedW = layer_type(stack_dims[0][0], stack_dims[0][1], weight=False, bias=self.opt['gcn_bias'], **layer_kwargs)
                    #insert parameter
                    GCN_fixedW.symm_weight = nn.Parameter(torch.Tensor(stack_dims[0][0], stack_dims[0][1]))
                    init.xavier_uniform_(GCN_fixedW.symm_weight)
                    #check it is in the model.parameters
                else:
                    GCN_fixedW = layer_type(stack_dims[0][0], stack_dims[0][1], bias=self.opt['gcn_bias'],  **layer_kwargs)
            else:
                # GCN_fixedW = layer_type(stack_dims[1][0], stack_dims[1][1], **layer_kwargs)
                if self.opt['gcn_symm']:
                    GCN_fixedW = layer_type(stack_dims[1][0], stack_dims[1][1], weight=False, bias=self.opt['gcn_bias'], **layer_kwargs)
                    GCN_fixedW.symm_weight = nn.Parameter(torch.Tensor(stack_dims[1][0], stack_dims[1][1]))
                    init.xavier_uniform_(GCN_fixedW.symm_weight)
                else:
                    GCN_fixedW = layer_type(stack_dims[1][0], stack_dims[1][1], bias=self.opt['gcn_bias'], **layer_kwargs)


        for indx, (in_feat, out_feat) in enumerate(stack_dims):
            if self.opt['gcn_mid_dropout']:
                stack.append(nn.Dropout(dropout))

            if in_feat != out_feat: #overide to ignore fixed W or residual blocks if using a convolutional encoder/decoder
                #note can't have a symmetric W here
                stack.append(GraphConv(in_feat, out_feat, bias=self.opt['gcn_bias'], **layer_kwargs))
            elif self.opt['gcn_fixed']:
                stack.append(GCN_fixedW)
            else:
                if self.opt['gcn_symm']:
                    layerConv = layer_type(in_feat, out_feat, weight=False, bias=self.opt['gcn_bias'], **layer_kwargs)
                    layerConv.symm_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
                    init.xavier_uniform_(layerConv.symm_weight)
                else:
                    layerConv = layer_type(in_feat, out_feat, bias=self.opt['gcn_bias'], **layer_kwargs)
                # stack.append(layer_type(in_feat, out_feat, **layer_kwargs))
                stack.append(layerConv)

            if indx < len(stack_dims) - 1 or top_is_proj:
                if self.opt['gcn_non_lin']:
                    stack.append(self.nonlinearity())

        if self.opt['gcn_enc_dec']:
            stack.append(nn.Dropout(dropout))
            # stack.append(nn.Linear(dims[-1][0], dims[-1][1]))
            self.m2 = nn.Linear(dims[-1][0], dims[-1][1])

        return GraphSequential(stack, self.opt)

    def encoder(self, features, pos_encoding=None):
        if self.opt['gcn_enc_dec']:
            features = F.dropout(features, self.opt['input_dropout'], training=self.training)
            features = self.m1(features, pos_encoding=None)
        return features

    def forward_XN(self, graph, features):
        features = self.encoder(features, pos_encoding=None)
        features = self.gcn_stack(graph, features)
        return features

    def forward(self, features, pos_encoding): #run_GNN.py
        if self.opt['function'] in ['gcn_dgl', 'gcn_res_dgl']:
            graph = dgl.graph((self.edge_index[0], self.edge_index[1])).to(self.device)
        elif self.opt['function'] == 'gcn2':
            graph = self.edge_index

        if self.enable_gcn and self.enable_mlp:
            pass
            # features = self.mixing_coeffs[0] * self.gcn_stack(graph, features) + self.mixing_coeffs[0] * self.mlp_stack(
            #     graph, features)
        elif self.enable_gcn:

            features = self.forward_XN(graph, features)

            if self.opt['gcn_enc_dec']:
                self.m2 = nn.Linear(features)

        elif self.enable_mlp:
            pass
            # features = self.mlp_stack(graph, features)

        if self.top_is_proj:
            pass
            # if self.enable_gcn and self.enable_mlp:
            #     features = self.pre_top_proj_nonlinearity(features)
            # features = self.top_proj(features)

        return features