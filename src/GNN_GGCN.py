#code taken from https://github.com/Yujun-Yan/Heterophily_and_oversmoothing


import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm


# -------------------------------------------------------------------------------------------GGCN------------------------------------------------------------------------------------
class GGCNlayer_SP(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5,
                 deg_intercept_init=0.5):
        super(GGCNlayer_SP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5, 0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init, 0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0 * torch.ones([3]))
            self.adj_remove_diag = None
            if use_decay:
                self.scale = nn.Parameter(2 * torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init * torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)

    def precompute_adj_wo_diag(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_wo_diag_ind = (adj_i[0, :] != adj_i[1, :])
        self.adj_remove_diag = torch.sparse.FloatTensor(adj_i[:, adj_wo_diag_ind], adj_v[adj_wo_diag_ind], adj.size())

    def non_linear_degree(self, a, b, s):
        i = s._indices()
        v = s._values()
        return torch.sparse.FloatTensor(i, self.sftpls(a * v + b), s.size())

    def get_sparse_att(self, adj, Wh):
        i = adj._indices()
        Wh_1 = Wh[i[0, :], :]
        Wh_2 = Wh[i[1, :], :]
        sim_vec = F.cosine_similarity(Wh_1, Wh_2)
        sim_vec_pos = F.relu(sim_vec)
        sim_vec_neg = -F.relu(-sim_vec)
        return torch.sparse.FloatTensor(i, sim_vec_pos, adj.size()), torch.sparse.FloatTensor(i, sim_vec_neg,
                                                                                              adj.size())

    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.non_linear_degree(self.deg_coeff[0], self.deg_coeff[1], degree_precompute)

        Wh = self.fcn(h)
        if self.use_sign:
            if self.adj_remove_diag is None:
                self.precompute_adj_wo_diag(adj)
        if self.use_sign:
            e_pos, e_neg = self.get_sparse_att(adj, Wh)
            if self.use_degree:
                attention_pos = self.adj_remove_diag * sc * e_pos
                attention_neg = self.adj_remove_diag * sc * e_neg
            else:
                attention_pos = self.adj_remove_diag * e_pos
                attention_neg = self.adj_remove_diag * e_neg

            prop_pos = torch.sparse.mm(attention_pos, Wh)
            prop_neg = torch.sparse.mm(attention_neg, Wh)

            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale * (coeff[0] * prop_pos + coeff[1] * prop_neg + coeff[2] * Wh)

        else:
            if self.use_degree:
                prop = torch.sparse.mm(adj * sc, Wh)
            else:
                prop = torch.sparse.mm(adj, Wh)

            result = prop
        return result


class GGCNlayer(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5,
                 deg_intercept_init=0.5):
        super(GGCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5, 0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init, 0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0 * torch.ones([3]))
            if use_decay:
                self.scale = nn.Parameter(2 * torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init * torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)

    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.deg_coeff[0] * degree_precompute + self.deg_coeff[1]
            sc = self.sftpls(sc)

        Wh = self.fcn(h)
        if self.use_sign:
            prod = torch.matmul(Wh, torch.transpose(Wh, 0, 1))
            sq = torch.unsqueeze(torch.diag(prod), 1)
            scaling = torch.matmul(sq, torch.transpose(sq, 0, 1))
            e = prod / torch.max(torch.sqrt(scaling), 1e-9 * torch.ones_like(scaling))
            e = e - torch.diag(torch.diag(e))
            if self.use_degree:
                attention = e * adj * sc
            else:
                attention = e * adj

            attention_pos = F.relu(attention)
            attention_neg = -F.relu(-attention)
            prop_pos = torch.matmul(attention_pos, Wh)
            prop_neg = torch.matmul(attention_neg, Wh)

            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale * (coeff[0] * prop_pos + coeff[1] * prop_neg + coeff[2] * Wh)

        else:
            if self.use_degree:
                prop = torch.matmul(adj * sc, Wh)
            else:
                prop = torch.matmul(adj, Wh)

            result = prop

        return result


class GGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, decay_rate, exponent, use_degree=True, use_sign=True,
                 use_decay=True, use_sparse=False, scale_init=0.5, deg_intercept_init=0.5, use_bn=False, use_ln=False):
        super(GGCN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if use_sparse:
            model_sel = GGCNlayer_SP
        else:
            model_sel = GGCNlayer
        self.convs.append(model_sel(nfeat, nhidden, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        for _ in range(nlayers - 2):
            self.convs.append(
                model_sel(nhidden, nhidden, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        self.convs.append(model_sel(nhidden, nclass, use_degree, use_sign, use_decay, scale_init, deg_intercept_init))
        self.fcn = nn.Linear(nfeat, nhidden)
        self.act_fn = F.elu
        self.dropout = dropout
        self.use_decay = use_decay
        if self.use_decay:
            self.decay = decay_rate
            self.exponent = exponent
        self.degree_precompute = None
        self.use_degree = use_degree
        self.use_sparse = use_sparse
        self.use_norm = use_bn or use_ln
        if self.use_norm:
            self.norms = nn.ModuleList()
        if use_bn:
            for _ in range(nlayers - 1):
                self.norms.append(nn.BatchNorm1d(nhidden))
        if use_ln:
            for _ in range(nlayers - 1):
                self.norms.append(nn.LayerNorm(nhidden))

    def precompute_degree_d(self, adj):
        diag_adj = torch.diag(adj)
        diag_adj = torch.unsqueeze(diag_adj, dim=1)
        self.degree_precompute = diag_adj / torch.max(adj, 1e-9 * torch.ones_like(adj)) - 1

    def precompute_degree_s(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_diag_ind = (adj_i[0, :] == adj_i[1, :])
        adj_diag = adj_v[adj_diag_ind]
        v_new = torch.zeros_like(adj_v)
        for i in range(adj_i.shape[1]):
            v_new[i] = adj_diag[adj_i[0, i]] / adj_v[i] - 1
        self.degree_precompute = torch.sparse.FloatTensor(adj_i, v_new, adj.size())

    def forward(self, x, adj):
        if self.use_degree:
            if self.degree_precompute is None:
                if self.use_sparse:
                    self.precompute_degree_s(adj)
                else:
                    self.precompute_degree_d(adj)
        x = F.dropout(x, self.dropout, training=self.training)
        layer_previous = self.fcn(x)
        layer_previous = self.act_fn(layer_previous)
        layer_inner = self.convs[0](x, adj, self.degree_precompute)

        for i, con in enumerate(self.convs[1:]):
            if self.use_norm:
                layer_inner = self.norms[i](layer_inner)
            layer_inner = self.act_fn(layer_inner)
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if i == 0:
                layer_previous = layer_inner + layer_previous
            else:
                if self.use_decay:
                    coeff = math.log(self.decay / (i + 2) ** self.exponent + 1)
                else:
                    coeff = 1
                layer_previous = coeff * layer_inner + layer_previous
            layer_inner = con(layer_previous, adj, self.degree_precompute)
        return F.log_softmax(layer_inner, dim=1)
