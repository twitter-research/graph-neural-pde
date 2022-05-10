import time, datetime
import os
import argparse
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import homophily, add_remaining_self_loops, to_undirected
from torch_scatter import scatter_add
import torch.nn.functional as F
import wandb
# from ogb.nodeproppred import Evaluator

from GNN import GNN
from GNN_early import GNNEarly
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
from GNN_GCN import GCN, MLP
from GNN_GCNMLP import GNNMLP
from data import get_dataset, set_train_val_test_split
from graph_rewiring import apply_KNN, apply_beltrami, apply_edge_sampling, dirichlet_energy
from best_params import best_params_dict
from greed_params import greed_test_params, greed_run_params, greed_hyper_params, greed_ablation_params, tf_ablation_args, not_sweep_args
from reports import reports_manager #run_reports, run_reports_lie_trotter, reports_manager
from heterophilic import get_fixed_splits
from data_synth_hetero import get_pyg_syn_cora

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
    lf = torch.nn.functional.nll_loss if model.opt['dataset'] == 'ogbn-arxiv' else torch.nn.CrossEntropyLoss()

    if model.opt['wandb_watch_grad']:  # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(model, lf, log="all", log_freq=10)

    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
    else:
        train_pred_idx = data.train_mask

    out = model(feat, pos_encoding)

    if model.opt['dataset'] == 'ogbn-arxiv':
        # lf = torch.nn.functional.nll_loss
        loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        # lf = torch.nn.CrossEntropyLoss()
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
    # torch.autograd.set_detect_anomaly(True)
    loss.backward()  # retain_graph=True)
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()

    return loss.item()


def train_OGB(model, mp, optimizer, data, pos_encoding=None):
    lf = torch.nn.functional.nll_loss if model.opt['dataset'] == 'ogbn-arxiv' else torch.nn.CrossEntropyLoss()

    if model.opt['wandb_watch_grad']:  # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(model, lf, log="all", log_freq=10)

    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
    else:
        train_pred_idx = data.train_mask

    pos_encoding = mp(pos_encoding).to(model.device)
    out = model(feat, pos_encoding)

    if model.opt['dataset'] == 'ogbn-arxiv':
        # lf = torch.nn.functional.nll_loss
        loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        # lf = torch.nn.CrossEntropyLoss()
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
    epoch = model.odeblock.odefunc.epoch
    if opt['wandb']:
        # wandb tracking
        # need to calc loss again
        lf = torch.nn.CrossEntropyLoss()
        loss = lf(logits[data.train_mask], data.y.squeeze()[data.train_mask])
        wandb_log(data, model, opt, loss, accs[0], accs[1], accs[2], epoch)
        model.odeblock.odefunc.wandb_step = 0  # resets the wandbstep counter in function after eval forward pass

    if opt['run_track_reports'] and epoch in opt['wandb_epoch_list']:
        reports_manager(model, data)

    return accs

@torch.no_grad()
def calc_energy_homoph(data, model, opt):
    # every epoch stats for greed linear and non linear
    num_nodes = data.num_nodes

    x0 = model.encoder(data.x)
    T0_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, x0)

    x0r = x0 / torch.norm(x0, p='fro')
    T0r_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, x0r)

    xN = model.forward_XN(data.x)
    TN_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, xN)

    xNr = xN / torch.norm(xN, p='fro')
    TNr_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, xNr)

    enc_pred = model.m2(x0).max(1)[1]
    if opt['lie_trotter'] == 'gen_2':
        if model.odeblock.funcs[-1].opt['lt_block_type'] == 'label':
            logits = model.odeblock.odefunc.GNN_postXN(xN)
            pred = logits.max(1)[1]
        else:
            pred = model.m2(xN).max(1)[1]
    else:
        pred = model.m2(xN).max(1)[1]
    enc_pred_homophil = homophily(edge_index=data.edge_index, y=enc_pred)
    pred_homophil = homophily(edge_index=data.edge_index, y=pred)
    label_homophil = homophily(edge_index=data.edge_index, y=data.y)
    return T0_dirichlet, T0r_dirichlet, TN_dirichlet, TNr_dirichlet, enc_pred_homophil, pred_homophil, label_homophil

@torch.no_grad()
def wandb_log(data, model, opt, loss, train_acc, val_acc, test_acc, epoch):
    model.eval()

    if opt['function'] in ['gcn', 'mlp', 'gcn2', 'gcn_dgl', 'gcn_res_dgl']:
        wandb.log({"loss": loss,
                   "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
                   "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                   "epoch_step": epoch})
        return

    # every epoch stats for greed linear and non linear
    num_nodes = data.num_nodes
    x0 = model.encoder(data.x)
    edges = torch.cat([model.odeblock.odefunc.edge_index, model.odeblock.odefunc.self_loops], dim=1)
    xN = model.forward_XN(data.x)
    T0_dirichlet, T0r_dirichlet, TN_dirichlet, TNr_dirichlet, enc_pred_homophil, pred_homophil, label_homophil = calc_energy_homoph(data, model, opt)

    if opt['function'] == "greed_linear_hetero":
        LpR = model.odeblock.odefunc.L_0 + model.odeblock.odefunc.R_0
        T0_dirichlet_W = dirichlet_energy(edges, LpR, num_nodes, x0)
        TN_dirichlet_W = dirichlet_energy(edges, LpR, num_nodes, xN)

        if opt['diffusion']:
            a = model.odeblock.odefunc.mean_attention_0
            a_row_max = scatter_add(a, model.odeblock.odefunc.edge_index[0], dim=0, dim_size=num_nodes).max()
            a_row_min = scatter_add(a, model.odeblock.odefunc.edge_index[0], dim=0, dim_size=num_nodes).min()
        else:
            a_row_max = 0
            a_row_min = 0

        if opt['repulsion']:
            b = model.odeblock.odefunc.mean_attention_R0
            b_row_max = scatter_add(b, model.odeblock.odefunc.edge_index[0], dim=0, dim_size=num_nodes).max()
            b_row_min = scatter_add(b, model.odeblock.odefunc.edge_index[0], dim=0, dim_size=num_nodes).min()
        else:
            b_row_max = 0
            b_row_min = 0

        if opt['alpha_style'] == 'diag':
            alpha = model.odeblock.odefunc.alpha.mean()
        elif opt['alpha_style'] == 'free':
            alpha = model.odeblock.odefunc.alpha.data
        else:
            alpha = model.odeblock.odefunc.alpha

        if opt['wandb_track_grad_flow'] and epoch in opt['wandb_epoch_list']:
            pass
            # placeholder for if we need to apply the evolution visualisations to the linear case

        wandb.log({"loss": loss,
                   # "tmp_train_acc": tmp_train_acc, "tmp_val_acc": tmp_val_acc, "tmp_test_acc": tmp_test_acc,
                   "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
                   "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                   "T0_dirichlet": T0_dirichlet, "TN_dirichlet": TN_dirichlet,
                   "T0_dirichlet_W": T0_dirichlet_W, "TN_dirichlet_W": TN_dirichlet_W,
                   "enc_pred_homophil": enc_pred_homophil, "pred_homophil": pred_homophil,
                   "label_homophil": label_homophil,
                   "a_row_max": a_row_max, "a_row_min": a_row_min, "b_row_max": b_row_max, "b_row_min": b_row_min,
                   "alpha": alpha,
                   "epoch_step": epoch})

    elif opt['function'] == "greed_non_linear":
        print(f"epoch {epoch}, delta: {model.odeblock.odefunc.delta.detach()}, mu: {model.odeblock.odefunc.mu}, epsilon: {model.odeblock.odefunc.om_W_eps}")  # , nu: {model.odeblock.odefunc.om_W_nu}")

        wandb.log({"loss": loss,
                   # "tmp_train_acc": tmp_train_acc, "tmp_val_acc": tmp_val_acc, "tmp_test_acc": tmp_test_acc,
                   "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
                   "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                   "T0_dirichlet": T0_dirichlet, "TN_dirichlet": TN_dirichlet,
                   "enc_pred_homophil": enc_pred_homophil, "pred_homophil": pred_homophil,
                   "label_homophil": label_homophil, "delta": model.odeblock.odefunc.delta.detach(),
                   "drift_eps": model.odeblock.odefunc.drift_eps.detach() if opt['drift'] else 0,
                   "W_rank": torch.matrix_rank(model.odeblock.odefunc.gnl_W.detach()),
                   # "a_row_max": a_row_max, "a_row_min": a_row_min,
                   "epoch_step": epoch})
    else:
        wandb.log({"loss": loss,
                   "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
                   "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                   "epoch_step": epoch})


def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)


@torch.no_grad()
def test_OGB(model, data, pos_encoding, opt):
    pass
    # if opt['dataset'] == 'ogbn-arxiv':
    #     name = 'ogbn-arxiv'
    #
    # feat = data.x
    # if model.opt['use_labels']:
    #     feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    #
    # evaluator = Evaluator(name=name)
    # model.eval()
    #
    # out = model(feat, pos_encoding).log_softmax(dim=-1)
    # y_pred = out.argmax(dim=-1, keepdim=True)
    #
    # train_acc = evaluator.eval({
    #     'y_true': data.y[data.train_mask],
    #     'y_pred': y_pred[data.train_mask],
    # })['acc']
    # valid_acc = evaluator.eval({
    #     'y_true': data.y[data.val_mask],
    #     'y_pred': y_pred[data.val_mask],
    # })['acc']
    # test_acc = evaluator.eval({
    #     'y_true': data.y[data.test_mask],
    #     'y_pred': y_pred[data.test_mask],
    # })['acc']
    #
    # return train_acc, valid_acc, test_acc


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
    # gnl_W_style: diag_dom, diag
    # gnl_W_diag_init: uniform, uniform
    # gnl_W_param_free: True, False
    # gnl_omega: diag, diag
    # gnl_omega_diag: free, free
    # use_mlp: False, True
    # test_mu_0: True, True
    # add_source: True, True
    wandb.config.update({'gnl_W_style': opt['greed_params'][0]}, allow_val_change=True)
    wandb.config.update({'gnl_W_diag_init': opt['greed_params'][1]}, allow_val_change=True)
    wandb.config.update({'gnl_W_param_free': opt['greed_params'][2]}, allow_val_change=True)
    wandb.config.update({'gnl_omega': opt['greed_params'][3]}, allow_val_change=True)
    wandb.config.update({'gnl_omega_diag': opt['greed_params'][4]}, allow_val_change=True)
    wandb.config.update({'use_mlp': opt['greed_params'][5]}, allow_val_change=True)
    wandb.config.update({'test_mu_0': opt['greed_params'][6]}, allow_val_change=True)
    wandb.config.update({'add_source': opt['greed_params'][7]}, allow_val_change=True)

def main(cmd_opt):
    if cmd_opt['use_best_params']:
        best_opt = best_params_dict[cmd_opt['dataset']]
        opt = {**cmd_opt, **best_opt}
        merge_cmd_args(cmd_opt, opt)
    else:
        opt = cmd_opt

    if opt['wandb']:
        if opt['use_wandb_offline']:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "run"
    else:
        os.environ["WANDB_MODE"] = "disabled"  # sets as NOOP, saves keep writing: if opt['wandb']:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device

    if 'wandb_run_name' in opt.keys():
        wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                               name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True)
    else:
        wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                               reinit=True, config=opt,
                               allow_val_change=True)  # required when update hidden_dim in beltrami

    # wandb.config.update(opt, allow_val_change=True) #required when update hidden_dim in beltrami
    opt = wandb.config  # access all HPs through wandb.config, so logging matches execution!
    if opt['gcn_params']: #temp function for GCN ablation
        unpack_gcn_params(opt)
    if opt['greed_params']: #temp function for GCN ablation
        unpack_greed_params(opt)

    wandb.define_metric("epoch_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
    if opt['wandb_track_grad_flow']:
        wandb.define_metric("grad_flow_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
        wandb.define_metric("gf_e*", step_metric="grad_flow_step")  # grad_flow_epoch*

    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    if opt['dataset'] in ['chameleon','squirrel','other hetero?']: #todo put this in data loader
        ### added self loops and make undirected for chameleon
        dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    if opt['beltrami']:
        pos_encoding = apply_beltrami(dataset.data, opt).to(device)
        opt['pos_enc_dim'] = pos_encoding.shape[1]
    else:
        pos_encoding = None

    this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test

    results = []
    for rep in range(opt['num_splits']):
        print(f"rep {rep}")
        if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
            dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                                    num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)
        if opt['geom_gcn_splits']:
            if opt['dataset'] == "Citeseer":
                dataset = get_dataset(opt, '../data', opt['not_lcc']) #geom-gcn citeseer uses splits over LCC and not_LCC so need to repload each split
            data = get_fixed_splits(dataset.data, opt['dataset'], rep)
            dataset.data = data
        if opt['dataset'] == 'syn_cora':
            dataset = get_pyg_syn_cora("../data", opt, rep=rep+1)

        data = dataset.data.to(device)

        if opt['rewire_KNN'] or opt['fa_layer']:
            model = GNN_KNN(opt, dataset, device).to(device) if opt["no_early"] else GNNKNNEarly(opt, dataset,
                                                                                                 device).to(device)
        elif opt['function'] in ['gcn']:
            model = GCN(opt, dataset, hidden=[opt['hidden_dim']], dropout=opt['dropout'], device=device).to(device)
        elif opt['function'] in ['mlp']:
            model = MLP(opt, dataset, device=device).to(device)
        elif opt['function'] in ['gcn2', 'gcn_dgl', 'gcn_res_dgl']:
            hidden_feat_repr_dims = int(opt['time'] // opt['step_size']) * [opt['hidden_dim']]
            feat_repr_dims = [dataset.data.x.shape[1]] + hidden_feat_repr_dims + [dataset.num_classes]
            model = GNNMLP(opt, dataset, device, feat_repr_dims,
                           enable_mlp=True if opt['function']=='mlp' else False,
                           enable_gcn=True if opt['function'] in ['gcn2', 'gcn_dgl', 'gcn_res_dgl'] else False,
                           learnable_mixing=False, use_sage=False, use_gat=False, gat_num_heads=1,
                           top_is_proj=False, use_prelu=False, dropout=opt['dropout']
                           ).to(device)
        else:
            model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(
                device)

        parameters = [p for p in model.parameters() if p.requires_grad]
        print(opt)
        print_model_params(model)
        optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
        best_time = best_epoch = train_acc = val_acc = test_acc = 0
        if opt['patience'] is not None:
            patience_count = 0
        for epoch in range(1, opt['epoch']):
            start_time = time.time()
            if opt['function'] in ['greed', 'greed_linear', 'greed_linear_homo', 'greed_linear_hetero',
                                   'greed_non_linear', 'greed_lie_trotter']:
                model.odeblock.odefunc.epoch = epoch

            if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
                ei = apply_KNN(data, pos_encoding, model, opt)
                model.odeblock.odefunc.edge_index = ei

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
            if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
                best_epoch = epoch
                val_acc = model.odeblock.test_integrator.solver.best_val
                test_acc = model.odeblock.test_integrator.solver.best_test
                train_acc = model.odeblock.test_integrator.solver.best_train
                best_time = model.odeblock.test_integrator.solver.best_time

            print(f"Epoch: {epoch}, Runtime: {time.time() - start_time:.3f}, Loss: {loss:.3f}, "
                  f"forward nfe {model.fm.sum}, backward nfe {model.bm.sum}, "
                  f"tmp_train: {tmp_train_acc:.4f}, tmp_val: {tmp_val_acc:.4f}, tmp_test: {tmp_test_acc:.4f}, "
                  f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}")

            if np.isnan(loss):
                wandb_run.finish()
                break
            if opt['patience'] is not None:
                if patience_count >= opt['patience']:
                    wandb_run.finish()
                    break
        print(
            f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")
        if opt['function'] == 'greed_non_linear':
            T0_dirichlet, T0r_dirichlet, TN_dirichlet, TNr_dirichlet, enc_pred_homophil, pred_homophil, label_homophil = calc_energy_homoph(data, model, opt)
        if opt['num_splits'] > 1:
            if opt['function'] == 'greed_non_linear':
                results.append([test_acc, val_acc, train_acc,
                                T0_dirichlet.cpu().detach().numpy(), T0r_dirichlet.cpu().detach().numpy(),
                                TN_dirichlet.cpu().detach().numpy(), TNr_dirichlet.cpu().detach().numpy(),
                                enc_pred_homophil, pred_homophil, label_homophil])
            else:
                results.append([test_acc, val_acc, train_acc])

    if opt['num_splits'] > 1:
        if opt['function'] == 'greed_non_linear':
            test_acc_mean, val_acc_mean, train_acc_mean, \
            T0_dirichlet_mean, T0r_dirichlet_mean, TN_dirichlet_mean, TNr_dirichlet_mean,\
            enc_pred_homophil_mean, pred_homophil_mean, label_homophil_mean \
                = np.mean(results, axis=0) * 100
            test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

            wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                             'test_acc_std': test_acc_std,
                             'T0_dirichlet_mean': T0_dirichlet_mean, 'T0r_dirichlet_mean': T0r_dirichlet_mean,
                             'TN_dirichlet_mean': TN_dirichlet_mean, 'TNr_dirichlet_mean': TNr_dirichlet_mean,
                             'enc_pred_homophil_mean': enc_pred_homophil_mean, 'pred_homophil_mean': pred_homophil_mean, 'label_homophil_mean': label_homophil_mean}
        else:
            test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results, axis=0) * 100
            test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
            wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                             'test_acc_std': test_acc_std}
        wandb.log(wandb_results)
        print(wandb_results)

    wandb_run.finish()
    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                        help='use the 10 fixed splits from '
                             'https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                        help='the number of splits to repeat the results on')
    # GNN args
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--patience', type=int, default=None, help='set if training should use patience on val acc')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, help='laplacian, transformer, greed, GAT')
    # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
    #                     help='Add a fully connected layer to the encoder.')
    # parser.add_argument('--add_source', dest='add_source', action='store_true',
    #                     help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    # parser.add_argument("--no_early", action="store_true",
    #                     help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")

    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    # rewiring args
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                        help="obove this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                        help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    parser.add_argument('--att_samp_pct', type=float, default=1,
                        help="float in [0,1). The percentage of edges to retain based on attention scores")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                        help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                        help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
    parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
    parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
    parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
    parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
    parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
    parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
    parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
    parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
    parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
    # beltrami args
    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
    parser.add_argument('--pos_enc_type', type=str, default="DW64",
                        help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
    parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
    parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                        help="random, ,anchored, importance, degree")
    parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
    parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--edge_sampling_space', type=str, default="attention",
                        help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
    # parser.add_argument('--symmetric_QK', action='store_true',
    #                     help='makes the attention symmetric for rewring in QK space')
    # parser.add_argument('--symmetric_attention', action='store_true',
    #                     help='makes the attention symmetric via (A+A.T)/2')#for rewring in QK space')
    # parser.add_argument('--sym_row_max', action='store_true',
    #                     help='makes every row sum less than 1 by dividing by max rum some')
    parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")
    parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

    # wandb logging and tuning
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('-wandb_offline', dest='use_wandb_offline',
                        action='store_true')  # https://docs.wandb.ai/guides/technical-faq

    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    parser.add_argument('--wandb_track_grad_flow', action='store_true')

    parser.add_argument('--wandb_entity', default="graph_neural_diffusion", type=str,
                        help="jrowbottomwnb, ger__man")  # not used as default set in web browser settings
    parser.add_argument('--wandb_project', default="greed", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_output_dir', default='./wandb_output',
                        help='folder to output results, images and model checkpoints')
    # parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    # replaces the above
    parser.add_argument('--wandb_epoch_list', nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 96, 128, 254],
                        help='list of epochs to log gradient flow, 1 based')
    parser.add_argument('--run_track_reports', action='store_true', help="run_track_reports")
    parser.add_argument('--save_wandb_reports', action='store_true', help="save_wandb_reports")
    parser.add_argument('--save_local_reports', action='store_true', help="save_local_reports")

# wandb setup sweep args
    parser.add_argument('--tau_reg', type=float, default=2)
    parser.add_argument('--test_mu_0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_no_chanel_mix', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_omit_metric_L', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_omit_metric_R', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_remove_tanh', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_symmetric', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_outside', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_linear_L0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_R1R2_0', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_grand_metric', type=str, default='True')  # action='store_true')
    parser.add_argument('--test_tau_ones', type=str, default='True')  # action='store_true')

    # Temp changing these to be strings so can tune over
    parser.add_argument('--use_mlp', type=str, default='False')  # action='store_true')
    parser.add_argument('--add_source', type=str, default='False',
                        help='If try get rid of alpha param and the beta*x0 source term')
    parser.add_argument('--no_early', type=str, default='False')  # action='store_true')
    parser.add_argument('--symmetric_QK', type=str, default='False',
                        help='makes the attention symmetric for rewring in QK space')
    parser.add_argument('--symmetric_attention', type=str, default='False',
                        help='makes the attention symmetric via (A+A.T)/2')  # for rewring in QK space')
    parser.add_argument('--sym_row_max', type=str, default='False',
                        help='makes every row sum less than 1 by dividing by max rum some')

    # greed args
    parser.add_argument('--use_best_params', action='store_true', help="flag to take the best BLEND params")
    parser.add_argument('--greed_momentum', action='store_true', help="flag to use momentum grad flow")
    parser.add_argument('--momentum_alpha', type=float, default=0.2, help="alpha to use in momentum grad flow")
    parser.add_argument('--dim_p_omega', type=int, default=16, help="inner dimension for Omega")
    parser.add_argument('--dim_p_w', type=int, default=16, help="inner dimension for W")
    parser.add_argument('--gamma_epsilon', type=float, default=0.01,
                        help="epsilon value used for numerical stability in get_gamma")

    parser.add_argument('--XN_no_activation', type=str, default='False',
                        help='whether to relu activate the terminal state')
    parser.add_argument('--m2_mlp', type=str, default='False', help='whether to use decoder mlp')
    parser.add_argument('--attention_activation', type=str, default='exponential',
                        help='[exponential, sigmoid] activations for the GRAM matrix')
    parser.add_argument('--attention_normalisation', type=str, default='sym_row_col',
                        help='[mat_row_max, sym_row_col, row_bottom, "best"] how to normalise')
    parser.add_argument('--T0term_normalisation', type=str, default='T0_identity',
                        help='[T0_symmDegnorm, T0_symmDegnorm, T0_identity] normalise T0 term')
    parser.add_argument('--laplacian_norm', type=str, default='lap_noNorm',
                        help='[lap_symmDegnorm, lap_symmRowSumnorm, lap_noNorm] how to normalise L')
    parser.add_argument('--R_T0term_normalisation', type=str, default='T0_identity',
                        help='[T0_symmDegnorm, T0_symmDegnorm, T0_identity] normalise T0 term')
    parser.add_argument('--R_laplacian_norm', type=str, default='lap_noNorm',
                        help='[lap_symmDegnorm, lap_symmRowSumnorm, lap_noNorm] how to normalise L')

    parser.add_argument('--alpha_style', type=str, default='', help='"sigmoid", "free", "forced", "matrix"')
    parser.add_argument('--fix_alpha', type=float, default=None, help='control balance between diffusion and repulsion')
    parser.add_argument('--diffusion', type=str, default='True', help='turns on diffusion')
    parser.add_argument('--repulsion', type=str, default='False', help='turns on repulsion')
    # parser.add_argument('--drift', type=str, default='False', help='turns on drift')
    parser.add_argument('--W_type', type=str, default='identity', help='identity, diag, full')
    parser.add_argument('--R_W_type', type=str, default='identity', help='for repulsion: identity, diag, full')
    parser.add_argument('--R_depon_A', type=str, default='', help='R dependancy in A')
    parser.add_argument('--W_beta', type=float, default=0.5, help='for cgnn Ws orthoganal update')
    parser.add_argument('--tau_residual', type=str, default='False', help='makes tau residual')

    parser.add_argument('--drift', type=str, default='False', help='turns on drift')
    parser.add_argument('--gnl_thresholding', type=str, default='False', help='turns on pseudo inverse thresholding')
    parser.add_argument('--lie_trotter', type=str, default=None, help='None, gen_0, gen_1, gen_2')

    # GCN ablation args
    parser.add_argument('--gcn_fixed', type=str, default='False', help='fixes layers in gcn')
    parser.add_argument('--gcn_enc_dec', type=str, default='False', help='uses encoder decoder with GCN')
    parser.add_argument('--gcn_non_lin', type=str, default='False', help='uses non linearity with GCN')
    parser.add_argument('--gcn_symm', type=str, default='False', help='make weight matrix in GCN symmetric')
    parser.add_argument('--gcn_bias', type=str, default='False', help='make GCN include bias')
    parser.add_argument('--gcn_mid_dropout', type=str, default='False', help='dropout between GCN layers')
    parser.add_argument('--gcn_params', nargs='+', default=None, help='list of args for gcn ablation')
    parser.add_argument('--gcn_params_idx', type=int, default=0, help='index to track GCN ablation')

    # greed non linear args
    parser.add_argument('--gnl_style', type=str, default='scaled_dot',
                        help='scaled_dot, softmax_attention, general_graph')
    parser.add_argument('--gnl_activation', type=str, default='identity', help='identity, sigmoid, ...')
    parser.add_argument('--gnl_measure', type=str, default='ones', help='ones, deg_poly, nodewise')
    parser.add_argument('--gnl_omega', type=str, default='zero', help='zero, diag, sum')
    parser.add_argument('--gnl_omega_diag', type=str, default='free', help='free, const')
    parser.add_argument('--gnl_omega_activation', type=str, default='identity', help='identity, exponential')
    parser.add_argument('--gnl_W_style', type=str, default='sum', help='sum, prod, GS, cgnn, diag_dom')
    parser.add_argument('--gnl_attention', type=str, default='False', help='turns on attention for gnl general graph')

    parser.add_argument('--k_blocks', type=int, default=5, help='k_blocks')
    parser.add_argument('--block_size', type=int, default=5, help='block_size')
    parser.add_argument('--k_diags', type=float, default=11, help='k_diags')
    parser.add_argument('--k_diag_pc', type=float, default=0.1, help='percentage or dims diagonal')
    parser.add_argument('--gnl_omega_params', nargs='+', default=None, help='list of Omega args for ablation')
    parser.add_argument('--gnl_W_params', nargs='+', default=None, help='list of W args for ablation')
    parser.add_argument('--gnl_W_diag_init', type=str, default='identity', help='init of diag elements [identity, uniform, linear]')
    parser.add_argument('--gnl_W_param_free', type=str, default='True', help='allow parameter to require gradient')
    parser.add_argument('--gnl_W_diag_init_q', type=float, default=1.0, help='slope of init of spectrum of W')
    parser.add_argument('--gnl_W_diag_init_r', type=float, default=0.0, help='intercept of init of spectrum of W')
    parser.add_argument('--two_hops', type=str, default='False', help='flag for 2-hop energy')
    parser.add_argument('--time_dep_w', type=str, default='False', help='Learn a time dependent potentials')
    parser.add_argument('--target_homoph', type=str, default='0.80', help='target_homoph for syn_cora [0.00,0.10,..,1.00]')
    parser.add_argument('--greed_params', nargs='+', default=None, help='list of args for focus models')

    args = parser.parse_args()
    opt = vars(args)

    if opt['function'] in ['greed', 'greed_scaledDP', 'greed_linear', 'greed_linear_homo', 'greed_linear_hetero',
                           'greed_non_linear', 'greed_lie_trotter', 'gcn', 'gcn2', 'mlp', 'gcn_dgl', 'gcn_res_dgl']:
        opt = greed_run_params(opt)  ###basic params for GREED


    if not opt['wandb_sweep']:  # sweeps are run from YAML config so don't need these
        opt = not_sweep_args(opt, project_name='greed_runs', group_name='testing')
        # this includes args for running locally - specified in YAML for tunes
        #  opt = greed_hyper_params(opt)
        # opt = greed_ablation_params(opt)

    # applied to both sweeps and not sweeps
    opt = tf_ablation_args(opt)
    main(opt)

# terminal commands for sweeps
# wandb sweep ../wandb_sweep_configs/greed_sweep_grid.yaml
# ./run_sweeps.sh XXX
# nohup ./run_sweeps.sh XXX &


# --dataset texas --geom_gcn_splits --num_splits 10 --epoch 2 --function greed --use_best_params --method euler --step_size 0.25
# --dataset texas --geom_gcn_splits --num_splits 10 --epoch 2 --function greed_lin_homo --beltrami --pos_enc_type GDC --method euler --step_size 0.25 --self_loop_weight 0
# --dataset Cora --epoch 100 --function greed_linear_homo --beltrami --pos_enc_type GDC --method euler --step_size 0.25 --self_loop_weight 0 --test_tau_symmetric True
# --dataset Cora --use_best_params --function greed_linear_homo

# --dataset Cora --block attention_greed --function laplacian_greed --use_best_params --symetric_QK True --method euler --step_size 0.5 --no_early True
# --dataset Cora --block attention_greed --function laplacian_greed --use_best_params --symmetric_QK True --method euler --step_size 0.5 --no_early True
# --method euler --step_size 0.5 --no_early True

# --dataset Cora --function greed_linear_homo --use_best_params --symmetric_QK True --method euler --step_size 0.5 --no_early True
# --dataset Cora --epoch 100 --function greed_linear_homo --beltrami --pos_enc_type GDC --method euler --step_size 0.25 --self_loop_weight 0 --test_tau_symmetric True

# --dataset Cora --block attention_greed --function laplacian_greed --use_best_params --symmetric_QK True --method euler --step_size 0.5 --no_early True --attention_activation sigmoid
# --dataset Cora --epoch 100 --function greed_linear_homo --method euler --step_size 0.25 --self_loop_weight 0 --test_tau_symmetric True --symmetric_QK True --symmetric_attention False --attention_activation sigmoid --attention_normalisation sym_row_col --test_tau_ones True --use_best_params --T0term_normalisation T0_identity --T1term_normalisation T1_noNorm

# --dataset Cora --use_best_params --function greed_linear_homo --attention_activation exponential --attention_normalisation none --T0term_normalisation T0_rowSum --laplacian_norm lap_symmAttM_RowSumnorm

# --dataset chameleon --function greed_non_linear --use_best_params
