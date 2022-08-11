
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, alpha=1., gamma=1., dt=1.):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.emb = nn.Identity(nfeat,nhid)
        # self.emb = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.readout = nn.Linear(nhid,nclass)
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        ######Overidding channel mixing
        # self.lin = torch.nn.Identity(nfeat)

    def forward(self, data):
        edge_index = data.edge_index
        input = F.dropout(data.x, self.dropout, training=self.training)
        X = self.emb(input.unsqueeze(-1)) #initial speed
        X_all = [X]
        for i in range(self.nlayers):
            X = torch.relu(self.conv(X, edge_index))
            # X = X + self.dt * torch.relu(self.conv(X, edge_index))
            # X = X + self.dt * self.conv(X, edge_index) #torch.relu(self.conv(X, edge_index))
            X_all.append(X)
        X = F.dropout(X, self.dropout, training=self.training)
        out = self.readout(X)
        X_all = torch.stack(X_all,dim=1).squeeze(-1)   ## has shape n_nodes x (n_layers+1)
        return out, None, X_all

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, alpha=1., gamma=1., dt=1.):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.emb = nn.Identity(nfeat,nhid)
        # self.emb = nn.Linear(nfeat,nhid)
        self.conv = GATConv(nhid, nhid)
        self.readout = nn.Linear(nhid,nclass)
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, data):
        edge_index = data.edge_index
        input = F.dropout(data.x, self.dropout, training=self.training)
        X = self.emb(input.unsqueeze(-1)) #initial speed
        X_all = [X]
        for i in range(self.nlayers):
            X = torch.relu(self.conv(X, edge_index))
            # X = X + self.dt * torch.relu(self.conv(X, edge_index))
            X_all.append(X)
        X = F.dropout(X, self.dropout, training=self.training)
        out = self.readout(X)
        X_all = torch.stack(X_all,dim=1).squeeze(-1)   ## has shape n_nodes x (n_layers+1)
        return out, None, X_all

class GraphCON_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, alpha=1., gamma=1., dt=1.):
        super(GraphCON_GCN, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.emb = nn.Identity(nfeat,nhid)
        # self.emb = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.readout = nn.Linear(nhid,nclass)
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, data):
        edge_index = data.edge_index
        input = F.dropout(data.x, self.dropout, training=self.training)
        Y = self.emb(input.unsqueeze(-1)) #initial speed
        X = Y # initial position

        Y_all = [Y]
        X_all = [X]

        for i in range(self.nlayers):
            Y = Y + self.dt * (torch.relu(self.conv(X, edge_index)) - self.alpha * Y - self.gamma * X)
            X = X + self.dt * Y

            Y_all.append(Y)
            X_all.append(X)

        X = F.dropout(X, self.dropout, training=self.training)
        out = self.readout(X)

        Y_all = torch.stack(Y_all, dim=1).squeeze(-1)  ## has shape n_nodes x (n_layers+1)
        X_all = torch.stack(X_all,dim=1).squeeze(-1)   ## has shape n_nodes x (n_layers+1)

        return out, X_all, Y_all


class GraphCON_GCN_f1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, alpha=1., gamma=1., dt=1.):
        super(GraphCON_GCN_f1, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.emb = nn.Identity(nfeat,nhid)
        # self.emb = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.readout = nn.Linear(nhid,nclass)
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, data):
        edge_index = data.edge_index
        input = F.dropout(data.x, self.dropout, training=self.training)
        Y = self.emb(input) #initial speed
        X = Y # initial position
        # input = data.x #F.dropout(data.x, self.dropout, training=self.training)
        # Y = input#.unsqueeze(-1) #self.emb(input.unsqueeze(-1)) #initial speed

        Y_all = [Y]
        X_all = [X]
        # print(f"X0 {X}")

        for i in range(self.nlayers):
            Y = Y + self.dt * (torch.relu(self.conv(X, edge_index)) - self.alpha * Y - self.gamma * X)
            X = X + self.dt * Y

            Y_all.append(Y)
            X_all.append(X)

        X = F.dropout(X, self.dropout, training=self.training)
        out = self.readout(X)

        Y_all = torch.stack(Y_all, dim=1).squeeze(-1)  ## has shape n_nodes x (n_layers+1)
        X_all = torch.stack(X_all,dim=1).squeeze(-1)   ## has shape n_nodes x (n_layers+1)

        return out, X_all, Y_all


class GraphCON_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, alpha=1., gamma=1., dt=1., nheads=4):
        super(GraphCON_GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.emb = nn.Identity(nfeat,nhid)
        # self.emb = nn.Linear(nfeat,nhid)
        self.conv = self.conv = GATConv(nhid, nhid, heads=nheads)
        self.readout = nn.Linear(nhid,nclass)
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, data):
        edge_index = data.edge_index
        n_nodes = data.x.size(0)
        input = F.dropout(data.x, self.dropout, training=self.training)
        Y = self.emb(input.unsqueeze(-1))
        X = Y

        Y_all = [Y]
        X_all = [X]

        for i in range(self.nlayers):
            Y = Y + self.dt * (F.elu(self.conv(X, edge_index)).view(n_nodes, -1, self.nheads).mean(dim=-1) - self.alpha * Y - self.gamma * X)
            X = X + self.dt * Y

            Y_all.append(Y)
            X_all.append(X)

        X = F.dropout(X, self.dropout, training=self.training)
        out = self.readout(X)

        Y_all = torch.stack(Y_all, dim=1).squeeze(-1)  ## has shape n_nodes x (n_layers+1)
        X_all = torch.stack(X_all,dim=1).squeeze(-1)   ## has shape n_nodes x (n_layers+1)

        return out, X_all, Y_all


#TransformerConv(in_channels, out_channels, heads=1, concat=False, beta= False, dropout=0, edge_dim=None, bias=False, root_weight=False)

class GRAND(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, alpha=1., gamma=1., dt=1.):
        super(GRAND, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.emb = nn.Identity(nfeat,nhid)
        # self.emb = nn.Linear(nfeat,nhid)
        # self.conv = TransformerConv(nhid, nhid, heads=1, concat=False, beta= 0, dropout=0, edge_dim=None, bias=False, root_weight=False)
        self.conv = GRAND_conv(nhid, nhid, heads=1, concat=False, beta= False, dropout=0, edge_dim=None, bias=False, root_weight=False)
        # self.conv.lin_value = nn.Identity(nhid, nhid)
        self.readout = nn.Linear(nhid,nclass)
        self.dt = dt
        # self.alpha = alpha
        # self.gamma = gamma

    def forward(self, data):
        edge_index = data.edge_index
        input = F.dropout(data.x, self.dropout, training=self.training)
        X = self.emb(input.unsqueeze(-1)) #initial speed
        X_all = [X]
        for i in range(self.nlayers):
            X = X + self.dt * (self.conv(X, edge_index) - X)   #Ahat = A - I
            X_all.append(X)
        X = F.dropout(X, self.dropout, training=self.training)
        out = self.readout(X)
        X_all = torch.stack(X_all,dim=1).squeeze(-1)   ## has shape n_nodes x (n_layers+1)
        return out, None, X_all


class GRAND_conv(TransformerConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=False, beta= 0, dropout=0, edge_dim=None, bias=False, root_weight=False):
        super(GRAND_conv, self).__init__(in_channels, out_channels, heads=1, concat=True, beta=False, dropout=0.0, edge_dim=None, bias=True, root_weight=False)
        self.lin_skip = nn.Identity(in_channels, heads * out_channels)
        self.lin_value = nn.Identity(in_channels, heads * out_channels) ######Overidding channel mixing