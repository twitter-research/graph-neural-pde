import time, datetime
import os
import argparse
import json
import numpy as np
import torch
from torch_geometric.utils import homophily, add_remaining_self_loops, to_undirected
import wandb

from GNN import GNN
from data import get_dataset, set_train_val_test_split
from heterophilic import get_fixed_splits
from data_synth_hetero import get_pyg_syn_cora

from best_params import best_params_dict
from greed_params import greed_test_params, greed_run_params, greed_hyper_params, greed_ablation_params, tf_ablation_args, not_sweep_args
from graff_params import graff_opt

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
    onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]  # convert mask to linear index
    onehot[idx, labels.squeeze()[idx]] = 1

    return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
    """
    when using labels as features need to split training nodes into training and prediction
    """
    if data.train_mask.dtype == torch.bool:
        idx = torch.where(data.train_mask)[0]
    else:
        idx = data.train_mask
    mask = torch.rand(idx.shape) < mask_rate
    train_label_idx = idx[mask]
    train_pred_idx = idx[~mask]
    return train_label_idx, train_pred_idx


def train(model, optimizer, data, pos_encoding=None):
    lf = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
    else:
        train_pred_idx = data.train_mask

    out = model(feat, pos_encoding)

    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

    if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
        reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
        regularization_coeffs = model.regularization_coeffs

        reg_loss = sum(
            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
        )
        loss = loss + reg_loss

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()

    return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
    model.eval()
    feat = data.x
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    logits, accs = model(feat, pos_encoding), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)

#todo check this when have sorted best params
def merge_cmd_args(cmd_opt, opt):
    if cmd_opt['beltrami']:
        opt['beltrami'] = True
    if cmd_opt['function'] is not None:
        opt['function'] = cmd_opt['function']
    if cmd_opt['block'] is not None:
        opt['block'] = cmd_opt['block']
    if cmd_opt['self_loop_weight'] is not None:
        opt['self_loop_weight'] = cmd_opt['self_loop_weight']
    if cmd_opt['method'] is not None:
        opt['method'] = cmd_opt['method']
    if cmd_opt['step_size'] != 1:
        opt['step_size'] = cmd_opt['step_size']
    if cmd_opt['time'] != 1:
        opt['time'] = cmd_opt['time']
    if cmd_opt['epoch'] != 100:
        opt['epoch'] = cmd_opt['epoch']
    if cmd_opt['num_splits'] != 1:
        opt['num_splits'] = cmd_opt['num_splits']
    if cmd_opt['attention_type'] != '':
        opt['attention_type'] = cmd_opt['attention_type']
    if cmd_opt['max_iters'] != 100:
        opt['max_iters'] = cmd_opt['max_iters']


def unpack_gcn_params(opt):
    'temp function to help ablation'
    wandb.config.update({'gcn_params_idx': opt['gcn_params'][0]}, allow_val_change=True)
    wandb.config.update({'function': opt['gcn_params'][1]}, allow_val_change=True)
    wandb.config.update({'gcn_enc_dec': opt['gcn_params'][2]}, allow_val_change=True)
    wandb.config.update({'gcn_fixed': opt['gcn_params'][3]}, allow_val_change=True)
    wandb.config.update({'gcn_symm': opt['gcn_params'][4]}, allow_val_change=True)
    wandb.config.update({'gcn_non_lin': opt['gcn_params'][5]}, allow_val_change=True)


def unpack_greed_params(opt):
    'temp function for "focus" models'
    # W_style: diag_dom, diag
    # W_diag_init: uniform, uniform
    # W_param_free: True, False
    # Omega_style: diag, diag
    # Omega_diag: free, free
    # use_mlp: False, True
    # test_mu_0: True, True
    # add_source: True, True
    wandb.config.update({'W_style': opt['graff_params'][0]}, allow_val_change=True)
    wandb.config.update({'W_diag_init': opt['graff_params'][1]}, allow_val_change=True)
    wandb.config.update({'W_param_free': opt['graff_params'][2]}, allow_val_change=True)
    wandb.config.update({'Omega': opt['graff_params'][3]}, allow_val_change=True)
    wandb.config.update({'Omega_diag': opt['graff_params'][4]}, allow_val_change=True)
    wandb.config.update({'use_mlp': opt['graff_params'][5]}, allow_val_change=True)
    wandb.config.update({'test_mu_0': opt['graff_params'][6]}, allow_val_change=True)
    wandb.config.update({'add_source': opt['graff_params'][7]}, allow_val_change=True)

def unpack_omega_params(self):
    'temp function to help ablation'
    wandb.config.update({'Omega': self.opt['Omega_params'][0]}, allow_val_change=True)
    wandb.config.update({'Omega_diag': self.opt['Omega_params'][1]}, allow_val_change=True)
    wandb.config.update({'Omega_diag_val': self.opt['Omega_params'][2]}, allow_val_change=True)

def main(cmd_opt):
    if cmd_opt['use_best_params']:
        best_opt = best_params_dict[cmd_opt['dataset']]
        opt = {**cmd_opt, **best_opt}
        merge_cmd_args(cmd_opt, opt)
    else:
        opt = cmd_opt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device

    if opt['gcn_params']: #temp function for GCN ablation
        unpack_gcn_params(opt)
    if opt['graff_params']: #temp function for GCN ablation
        unpack_greed_params(opt)

    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    ### todo put this in data loader
    if opt['dataset'] in ['chameleon','squirrel','other hetero?']:
        ###todo added self loops and make undirected for chameleon & squirrel
        if opt['greed_SL']:
            dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)
        if opt['greed_undir']:
            dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    pos_encoding = None
    this_test = test
    results = []
    for rep in range(opt['num_splits']):
        print(f"rep {rep}")
        if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
            dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                                    num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)
        if opt['geom_gcn_splits']:
            if opt['dataset'] == "Citeseer":
                dataset = get_dataset(opt, '../data', opt['not_lcc']) #geom-gcn citeseer uses splits over LCC and not_LCC so need to reload each rep/split
            data = get_fixed_splits(dataset.data, opt['dataset'], rep)
            dataset.data = data
        if opt['dataset'] == 'syn_cora':
            dataset = get_pyg_syn_cora("../data", opt, rep=rep+1)

        data = dataset.data.to(device)
        model = GNN(opt, dataset, device).to(device)

        parameters = [p for p in model.parameters() if p.requires_grad]
        print(opt)
        print_model_params(model)
        optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
        best_time = best_epoch = train_acc = val_acc = test_acc = 0
        if opt['patience'] is not None:
            patience_count = 0
        for epoch in range(1, opt['epoch']):
            start_time = time.time()
            loss = train(model, optimizer, data, pos_encoding)
            tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

            best_time = opt['time']
            if tmp_val_acc > val_acc:
                best_epoch = epoch
                train_acc = tmp_train_acc
                val_acc = tmp_val_acc
                test_acc = tmp_test_acc
                best_time = opt['time']
                patience_count = 0
            else:
                patience_count += 1
            print(f"Epoch: {epoch}, Runtime: {time.time() - start_time:.3f}, Loss: {loss:.3f}, "
                  f"forward nfe {model.fm.sum}, backward nfe {model.bm.sum}, "
                  f"tmp_train: {tmp_train_acc:.4f}, tmp_val: {tmp_val_acc:.4f}, tmp_test: {tmp_test_acc:.4f}, "
                  f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}")

            if np.isnan(loss):
                break
            if opt['patience'] is not None:
                if patience_count >= opt['patience']:
                    break
        print(
            f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")

        if opt['num_splits'] > 1:
            results.append([test_acc, val_acc, train_acc])

    if opt['num_splits'] > 1:
        test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
        results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                             'test_acc_std': test_acc_std}
        print(results)
        return test_acc_mean, val_acc_mean, train_acc_mean, test_acc_std
    else:
        return train_acc, val_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #run args
    parser.add_argument('--use_best_params', action='store_true', help="flag to take the best BLEND params")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--patience', type=int, default=None, help='set if training should use patience on val acc')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')

    # data args
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw', help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5, help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true', help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true', help='use the 10 fixed splits from https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1, help='the number of splits to repeat the results on')
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    #todo check these
    parser.add_argument('--hetero_SL', type=str, default='True', help='control self loops for Chameleon/Squirrel')
    parser.add_argument('--hetero_undir', type=str, default='True', help='control undirected for Chameleon/Squirrel')

    # GNN args
    parser.add_argument('--block', type=str, help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, help='laplacian, transformer, greed, GAT')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true', help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true', help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true', help='If try get rid of alpha param and the beta*x0 source term')
    parser.add_argument('--XN_activation', action='store_true', help='whether to relu activate the terminal state')
    parser.add_argument('--m2_mlp', action='store_true', help='whether to use decoder mlp')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true', help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=0.1, help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun", help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true', help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1, help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0, help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000, help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--max_test_steps", type=int, default=100, help="Maximum number steps for the dopri5Early test integrator. used if getting OOM errors at test time")

    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")
    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    # GCN ablation args
    parser.add_argument('--gcn_fixed', type=str, default='False', help='fixes layers in gcn')
    parser.add_argument('--gcn_enc_dec', type=str, default='False', help='uses encoder decoder with GCN')
    parser.add_argument('--gcn_non_lin', type=str, default='False', help='uses non linearity with GCN')
    parser.add_argument('--gcn_symm', type=str, default='False', help='make weight matrix in GCN symmetric')
    parser.add_argument('--gcn_bias', type=str, default='False', help='make GCN include bias')
    parser.add_argument('--gcn_mid_dropout', type=str, default='False', help='dropout between GCN layers')
    parser.add_argument('--gcn_params', nargs='+', default=None, help='list of args for gcn ablation')
    parser.add_argument('--gcn_params_idx', type=int, default=0, help='index to track GCN ablation')

    # graff args
    parser.add_argument('--omega_style', type=str, default='zero', help='zero, diag')
    parser.add_argument('--omega_diag', type=str, default='free', help='free, const')
    parser.add_argument('--omega_params', nargs='+', default=None, help='list of Omega args for ablation')
    parser.add_argument('--w_style', type=str, default='sum', help='sum, prod, neg_prod, diag_dom, diag')
    parser.add_argument('--w_diag_init', type=str, default='identity', help='init of diag elements [identity, uniform, linear]')
    #todo boolean
    parser.add_argument('--w_param_free', type=str, default='True', help='allow parameter to require gradient')
    parser.add_argument('--w_diag_init_q', type=float, default=1.0, help='slope of init of spectrum of W')
    parser.add_argument('--w_diag_init_r', type=float, default=0.0, help='intercept of init of spectrum of W')
    parser.add_argument('--w_params', nargs='+', default=None, help='list of W args for ablation')
    #todo boolean
    parser.add_argument('--time_dep_w', type=str, default='False', help='Learn a time dependent potentials')
    parser.add_argument('--time_dep_struct_w', type=str, default='False', help='Learn a structured time dependent potentials')
    parser.add_argument('--graff_params', nargs='+', default=None, help='list of args for focus models')

    args = parser.parse_args()
    opt = vars(args)
    opt = graff_opt(opt)
    main(opt)