import time, datetime
import multiprocessing as mp
import os
import argparse
import json
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import homophily, add_remaining_self_loops, to_undirected, is_undirected, contains_self_loops, degree
from torch_scatter import scatter_add
import torch.nn.functional as F
import wandb

from GNN import GNN
from GNN_early import GNNEarly
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
from GNN_GCN import GCN, MLP, GAT
from GNN_GCNMLP import GNNMLP
from data import get_dataset, set_train_val_test_split
from graph_rewiring import apply_KNN, apply_beltrami, apply_edge_sampling
from utils import dirichlet_energy
from best_params import best_params_dict
from greed_params import greed_test_params, greed_run_params, greed_hyper_params, tf_ablation_args, not_sweep_args, zinc_params
from greed_reporting_fcts import calc_energy_homoph
from graff_params import hetero_params
from reports import reports_manager #run_reports, run_reports_lie_trotter, reports_manager
from heterophilic import get_fixed_splits
from data_synth_hetero import get_pyg_syn_cora


def get_zinc_data(split):
  path = '../data/ZINC'
  dataset = ZINC(path, subset=True, split=split)
  # dataset.num_classes = 1    #can't override existing value which is unique float values
  dataset.num_nodes = dataset.data.x.shape[0]
  dataset.data.edge_attr = None #not using edge features
  dataset.data.x = dataset.data.x.float()
  return dataset

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


def zinc_batch_reset(model, data):
    # things need to reset because number of nodes in each graph/batch is different

    # data.edge_index, _ = add_remaining_self_loops(data.edge_index) #add self loops to replicate naive GCN

    model.odeblock.odefunc.data = data
    model.odeblock.odefunc.edge_index = data.edge_index
    model.num_nodes = data.num_nodes
    model.odeblock.odefunc.n_nodes = data.num_nodes
    model.odeblock.odefunc.degree = degree(data.edge_index[0], data.num_nodes).to(model.device)
    model.odeblock.odefunc.self_loops = model.odeblock.odefunc.get_self_loops().to(model.device)
    model.odeblock.odefunc.deg_inv_sqrt = model.odeblock.odefunc.get_deg_inv_sqrt(data).to(model.device)
    model.odeblock.odefunc.deg_inv = model.odeblock.odefunc.deg_inv_sqrt * model.odeblock.odefunc.deg_inv_sqrt


def train(model, optimizer, train_loader, pos_encoding=None):
    lf = torch.nn.L1Loss()

    if model.opt['wandb_watch_grad']:  # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(model, lf, log="all", log_freq=10)

    model.train()
    train_error = 0
    cum_loss = 0
    for i, data in enumerate(train_loader):
        if model.opt['test_batches'] is not None:
            if i >= model.opt['test_batches']:
                break

        #things need to reset because number of nodes in each batch is different
        data = data.to(model.device)
        zinc_batch_reset(model, data)

        optimizer.zero_grad()
        feat = data.x
        out = model(feat, pos_encoding).squeeze()

        train_error += (out - data.y).abs().sum().item()
        loss = lf(out, data.y)
        if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
            reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
            regularization_coeffs = model.regularization_coeffs
            reg_loss = sum(reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0)
            loss = loss + reg_loss

        model.fm.update(model.getNFE())
        model.resetNFE()
        loss.backward()  # retain_graph=True)
        # #sum([1 if param.grad.sum() > 0 else 0 for param in optimizer.param_groups[0]['params']])
        # optimizer.param_groups[0]['params'][0].grad.sum()
        # sum([0 if param.grad is None else param.grad.sum() > 0 for param in optimizer.param_groups[0]['params']])
        optimizer.step() #[0 if param.grad is None else param.grad.sum() for param in optimizer.param_groups[0]['params']]
        model.bm.update(model.getNFE())
        model.resetNFE()

        cum_loss += loss.detach().item() * (data.batch.max() + 1)
    train_acc = train_error / len(train_loader.dataset)
    av_loss = cum_loss / len(train_loader.dataset)

    return av_loss.item(), train_acc


@torch.no_grad()
def test(model, loader):
    lf = torch.nn.L1Loss()

    model.eval()
    error = 0
    cum_loss = 0
    for i, data in enumerate(loader):
        if model.opt['test_batches'] is not None:
            if i >= model.opt['test_batches']:
                break

        # things need to reset because number of nodes in each batch is different
        data = data.to(model.device)
        zinc_batch_reset(model, data)

        out = model(data.x).squeeze()
        error += (out - data.y).abs().sum().item()
        cum_loss += lf(out, data.y) * (data.batch.max() + 1)

    return cum_loss / len(loader.dataset), error / len(loader.dataset)


@torch.no_grad()
def wandb_log(data, model, opt, loss, train_acc, val_acc, test_acc, epoch):
    model.eval()

    if opt['function'] in ['gcn', 'mlp', 'gcn2', 'gat_dgl']: #removed 'gcn_dgl' 'gcn_res_dgl', for energy ablation rebuttal
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
    T0_DE, T0r_DE, TN_DE, TNr_DE, T0_WDE, TN_WDE, enc_pred_homophil, pred_homophil, label_homophil = calc_energy_homoph(data, model, opt)

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

        T0_DE, T0r_DE, TN_DE, TNr_DE, T0_WDE, TN_WDE
        wandb.log({"loss": loss,
                   # "tmp_train_acc": tmp_train_acc, "tmp_val_acc": tmp_val_acc, "tmp_test_acc": tmp_test_acc,
                   "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
                   "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                   "T0_DE": T0_DE, "T0r_DE": T0r_DE,
                   "TN_DE": TN_DE, "TNr_DE": TNr_DE,
                   "T0_DE_W": T0_WDE, "TN_DE_W": TN_WDE,
                   "enc_pred_homophil": enc_pred_homophil, "pred_homophil": pred_homophil,
                   "label_homophil": label_homophil,
                   "a_row_max": a_row_max, "a_row_min": a_row_min, "b_row_max": b_row_max, "b_row_min": b_row_min,
                   "alpha": alpha,
                   "epoch_step": epoch})

    elif opt['function'] == "greed_non_linear":
        print(f"epoch {epoch}, delta: {model.odeblock.odefunc.delta.detach()}, mu: {model.odeblock.odefunc.mu}, epsilon: {model.odeblock.odefunc.om_W_eps}")  # , nu: {model.odeblock.odefunc.om_W_nu}")

        # if model.opt['gnl_W_style'] == 'loss_W_orthog':
        W_evec = model.odeblock.odefunc.W_evec
        loss_orthog = torch.pow(torch.norm(W_evec.T @ W_evec - torch.eye(model.hidden_dim, device=model.device), "fro"), 2)

        wandb.log({"loss": loss, "loss_orthog": loss_orthog,
                   # "tmp_train_acc": tmp_train_acc, "tmp_val_acc": tmp_val_acc, "tmp_test_acc": tmp_test_acc,
                   "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
                   "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                   "T0_DE": T0_DE, "T0r_DE": T0r_DE,
                   "TN_DE": TN_DE, "TNr_DE": TNr_DE,
                   "T0_DE_W": T0_WDE, "TN_DE_W": TN_WDE,
                   "enc_pred_homophil": enc_pred_homophil, "pred_homophil": pred_homophil,
                   "label_homophil": label_homophil, "delta": model.odeblock.odefunc.delta.detach(),
                   "drift_eps": model.odeblock.odefunc.drift_eps.detach() if opt['drift'] else 0,
                   "W_rank": torch.matrix_rank(model.odeblock.odefunc.gnl_W.detach()),
                   # "a_row_max": a_row_max, "a_row_min": a_row_min,
                   "epoch_step": epoch})

    elif opt['function'] in ["gcn_dgl", "gcn_res_dgl"]:

        wandb.log({"loss": loss,
                   "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                   "T0_DE": T0_DE, "T0r_DE": T0r_DE,
                   "TN_DE": TN_DE, "TNr_DE": TNr_DE,
                   "T0_DE_W": T0_WDE, "TN_DE_W": TN_WDE,
                   "enc_pred_homophil": enc_pred_homophil, "pred_homophil": pred_homophil,
                   "label_homophil": label_homophil,
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

def unpack_W_params(self):
    'temp function to help ablation'
    wandb.config.update({'gnl_W_style': self.opt['gnl_W_params'][0]}, allow_val_change=True)
    wandb.config.update({'gnl_W_diag_init': self.opt['gnl_W_params'][1]}, allow_val_change=True)

def unpack_omega_params(self):
    'temp function to help ablation'
    wandb.config.update({'gnl_omega': self.opt['gnl_omega_params'][0]}, allow_val_change=True)
    wandb.config.update({'gnl_omega_diag': self.opt['gnl_omega_params'][1]}, allow_val_change=True)
    wandb.config.update({'gnl_omega_diag_val': self.opt['gnl_omega_params'][2]}, allow_val_change=True)
    wandb.config.update({'gnl_omega_activation': self.opt['gnl_omega_params'][3]}, allow_val_change=True)


def main(cmd_opt):
    # if cmd_opt['use_best_params']:
    #     best_opt = best_params_dict[cmd_opt['dataset']]
    #     opt = {**cmd_opt, **best_opt}
    #     merge_cmd_args(cmd_opt, opt)
    # else:
    opt = cmd_opt

    opt['dataset'] = "ZINC" #set by deafult

    if opt['wandb']:
        if opt['wandb_offline']:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "run"
    else:
        os.environ["WANDB_MODE"] = "disabled"  # sets as NOOP, saves keep writing: if opt['wandb']:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device

    if 'wandb_run_name' in opt.keys():
        wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                               name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True,
                               settings=wandb.Settings(start_method="fork"))
    else:
        wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                               reinit=True, config=opt, allow_val_change=True,
                               settings=wandb.Settings(start_method="fork"))

    opt = wandb.config  # access all HPs through wandb.config, so logging matches execution!

    if opt['gcn_params']: #temp function for GCN ablation
        unpack_gcn_params(opt)
    if opt['greed_params']: #temp function for GCN ablation
        unpack_greed_params(opt)
    if opt['W_params']: #temp function for ablation
        unpack_W_params(opt)
    if opt['omega_params']: #temp function for ablation
        unpack_omega_params(opt)

    wandb.define_metric("epoch_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
    wandb.define_metric("train_acc", step_metric="epoch_step")
    wandb.define_metric("val_acc", step_metric="epoch_step")
    wandb.define_metric("test_acc", step_metric="epoch_step")
    wandb.define_metric("T0_dirichlet", step_metric="epoch_step")
    wandb.define_metric("TN_dirichlet", step_metric="epoch_step")
    wandb.define_metric("enc_pred_homophil", step_metric="epoch_step")
    wandb.define_metric("pred_homophil", step_metric="epoch_step")
    wandb.define_metric("label_homophil", step_metric="epoch_step")

    if opt['wandb_track_grad_flow']:
        wandb.define_metric("grad_flow_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
        wandb.define_metric("gf_e*", step_metric="grad_flow_step")  # grad_flow_epoch*

    # dataset = get_dataset(opt, '../data', opt['not_lcc'])
    train_dataset = get_zinc_data('train')
    val_dataset = get_zinc_data('val')
    test_dataset = get_zinc_data('test')

    #todo this cause errors in collate when on GPU
    # num_workers = 4 if torch.cuda.is_available() else 0

    train_loader = DataLoader(train_dataset, batch_size=opt['batch'], shuffle=True)#, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt['batch'], shuffle=False)#, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=opt['batch'], shuffle=False)#, num_workers=num_workers, pin_memory=True)

    if opt['beltrami']:
        pos_encoding = apply_beltrami(dataset.data, opt).to(device)
        opt['pos_enc_dim'] = pos_encoding.shape[1]
    else:
        pos_encoding = None

    results = []
    for rep in range(opt['num_splits']):
        print(f"rep {rep}")
        # if opt['function'] in ['gcn']:
        #     model = GCN(opt, train_dataset, hidden=[opt['hidden_dim']], dropout=opt['dropout'], device=device).to(device)
        # else: #greed_non_linear
        model = GNN(opt, train_dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, train_dataset, device).to(device)

        parameters = [p for p in model.parameters() if p.requires_grad]
        print(opt)
        print_model_params(model)
        optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt['lr_reduce_factor'],
                                                               threshold=opt['lr_schedule_threshold'], patience=opt['lr_schedule_patience'],
                                                               verbose=True)

        best_time = best_epoch = 0
        train_acc = val_acc = test_acc = np.inf
        if opt['patience'] is not None:
            patience_count = 0
        for epoch in range(1, opt['epoch']):
            start_time = time.time()
            model.odeblock.odefunc.epoch = epoch

            loss, tmp_train_acc = train(model, optimizer, train_loader, pos_encoding)

            val_loss, tmp_val_acc = test(model, val_loader)
            test_loss, tmp_test_acc = test(model, test_loader)

            scheduler.step(val_loss)

            #sample data (first batch) to run all reports
            data = train_dataset.data
            data = data.to(model.device)
            if opt['wandb']:
                wandb_log(data, model, opt, loss, tmp_train_acc, tmp_val_acc, tmp_test_acc, epoch)
                model.odeblock.odefunc.wandb_step = 0  # resets the wandbstep counter in function after eval forward pass

            if opt['run_track_reports'] and epoch in opt['wandb_epoch_list']:
                reports_manager(model, data)

            best_time = opt['time']
            # if tmp_val_acc > val_acc:
            if tmp_val_acc < val_acc:
                best_epoch = epoch
                train_acc = tmp_train_acc
                val_acc = tmp_val_acc
                test_acc = tmp_test_acc
                best_time = opt['time']
                if opt['patience'] is not None:
                    patience_count = 0
            else:
                if opt['patience'] is not None:
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
                  f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}, LR: {optimizer.param_groups[0]['lr']}")

            if np.isnan(loss):
                wandb_run.finish()
                break
            if opt['patience'] is not None:
                if patience_count >= opt['patience']:
                    # wandb_run.finish()
                    break
            if optimizer.param_groups[0]['lr'] < opt['min_lr']:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

        print(f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")
        if opt['function'] in ['greed_non_linear', 'gcn_dgl', 'gcn_res_dgl']:
            T0_DE, T0r_DE, TN_DE, TNr_DE, T0_WDE, TN_WDE, enc_pred_homophil, pred_homophil, label_homophil = calc_energy_homoph(data, model, opt)
            # if model.opt['gnl_W_style'] == 'loss_W_orthog':
            W_evec = model.odeblock.odefunc.W_evec
            loss_orthog = torch.pow(torch.norm(W_evec.T @ W_evec - torch.eye(model.hidden_dim, device=model.device), "fro"), 2)

        if opt['num_splits'] > 1:
            if opt['function'] in ['greed_non_linear', 'gcn_dgl', 'gcn_res_dgl']:
                results.append([test_acc*100, val_acc*100, train_acc*100, loss, loss_orthog.cpu().detach().numpy(),
                                T0_DE.cpu().detach().numpy(), T0r_DE.cpu().detach().numpy(),
                                TN_DE.cpu().detach().numpy(), TNr_DE.cpu().detach().numpy(),
                                T0_WDE.cpu().detach().numpy(), TN_WDE.cpu().detach().numpy(),
                                enc_pred_homophil*100, pred_homophil*100, label_homophil*100])
            else:
                results.append([test_acc*100, val_acc*100, train_acc*100, loss])

    if opt['num_splits'] > 1:
        if opt['function']  in ['greed_non_linear', 'gcn_dgl', 'gcn_res_dgl']:
            test_acc_mean, val_acc_mean, train_acc_mean, loss_mean, loss_orthog_mean,\
            T0_DE_mean, T0r_DE_mean, TN_DE_mean, TNr_DE_mean, T0_WDE_mean, TN_WDE_mean,\
            enc_pred_homophil_mean, pred_homophil_mean, label_homophil_mean \
                = np.mean(results, axis=0)
            test_acc_std = np.sqrt(np.var(results, axis=0)[0])

            wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                             'test_acc_std': test_acc_std, 'loss_mean': loss_mean, "loss_orthog_mean": loss_orthog_mean,
                             'T0_DE_mean': T0_DE_mean, 'T0r_DE_mean': T0r_DE_mean,
                             'TN_DE_mean': TN_DE_mean, 'TNr_DE_mean': TNr_DE_mean,
                             'T0_WDE_mean': T0_WDE_mean, 'TN_WDE_mean': TN_WDE_mean,
                             'enc_pred_homophil_mean': enc_pred_homophil_mean, 'pred_homophil_mean': pred_homophil_mean, 'label_homophil_mean': label_homophil_mean}
        else:
            test_acc_mean, val_acc_mean, train_acc_mean, loss_mean = np.mean(results, axis=0)
            test_acc_std = np.sqrt(np.var(results, axis=0)[0])
            wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                             'test_acc_std': test_acc_std}
    else:
        if opt['function'] in ['greed_non_linear', 'gcn_dgl', 'gcn_res_dgl']:
            wandb_results = {'test_mean': test_acc*100, 'val_mean': val_acc*100, 'train_mean': train_acc*100,
                             'loss_mean': loss, 'loss_orthog_mean': loss_orthog,
                             'T0_DE_mean': T0_DE, 'T0r_DE_mean': T0r_DE,
                             'TN_DE_mean': TN_DE, 'TNr_DE_mean': TNr_DE,
                             'T0_WDE_mean': T0_WDE, 'TN_WDE_mean': TN_WDE,
                             'enc_pred_homophil_mean': enc_pred_homophil, 'pred_homophil_mean': pred_homophil, 'label_homophil_mean': label_homophil}
        else:
            wandb_results = {'test_mean': test_acc*100, 'val_mean': val_acc*100, 'train_mean': train_acc*100, 'loss_mean': loss}

    if opt['wandb']:
        wandb.log(wandb_results)
        wandb_run.finish()
    print(wandb_results)
    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_feat_norm', type=str, default='True',
                        help='use pre transform NormalizeFeatures')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, help='Weight of self-loops.')
    # parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--use_labels', type=str, default='False', help='Also diffuse labels')
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
    parser.add_argument('--wandb_offline', action='store_true')  # https://docs.wandb.ai/guides/technical-faq

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
    parser.add_argument('--time2', type=float, default=None, help='LT2 block2 - End time of ODE integrator.')
    parser.add_argument('--time3', type=float, default=None, help='LT2 block3 - End time of ODE integrator.')

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
    parser.add_argument('--gnl_W_diag_init', type=str, default='uniform', help='init of diag elements [identity, uniform, linear]')
    parser.add_argument('--gnl_W_param_free', type=str, default='True', help='allow parameter to require gradient')
    parser.add_argument('--gnl_W_diag_init_q', type=float, default=1.0, help='slope of init of spectrum of W')
    parser.add_argument('--gnl_W_diag_init_r', type=float, default=0.0, help='intercept of init of spectrum of W')
    parser.add_argument('--gnl_W_norm', type=str, default='False', help='divide W matrix by its spectral radius')
    parser.add_argument('--two_hops', type=str, default='False', help='flag for 2-hop energy')
    parser.add_argument('--time_dep_w', type=str, default=None, help='Learn a time dependent potentials for w')
    parser.add_argument('--time_dep_omega', type=str, default=None, help='Learn a time dependent potentials for omega')
    parser.add_argument('--time_dep_q', type=str, default=None, help='Learn a time dependent potentials for q')
    parser.add_argument('--num_lamb_w', type=int, default=1, help='number of time dep gaussians for w')
    parser.add_argument('--num_lamb_omega', type=int, default=1, help='number of time dep gaussians for omega')
    parser.add_argument('--num_lamb_q', type=int, default=1, help='number of time dep gaussians for q')
    parser.add_argument('--target_homoph', type=str, default='0.80', help='target_homoph for syn_cora [0.00,0.10,..,1.00]')
    parser.add_argument('--hetero_SL', type=str, default='', help='control self loops for Chameleon/Squirrel')
    parser.add_argument('--hetero_undir', type=str, default='', help='control undirected for Chameleon/Squirrel')
    parser.add_argument('--gnl_savefolder', type=str, default='', help='ie ./plots/{chamleon_gnlgraph_nodrift}')

    parser.add_argument('--omega_params', nargs='+', default=None, help='list of Omega args for ablation')
    parser.add_argument('--W_params', nargs='+', default=None, help='list of W args for ablation')
    parser.add_argument('--greed_params', nargs='+', default=None, help='list of args for focus models')

    parser.add_argument('--loss_reg', type=int, default=None, help='1-6')
    parser.add_argument('--loss_reg_weight', type=float, default=1.0, help='weighting for loss reg term')
    parser.add_argument('--loss_reg_delay', type=int, default=0.0, help='num epochs epochs to wait before applying loss reg')
    parser.add_argument('--loss_reg_certainty', type=float, default=1.0, help='amount of certainty to encode in prediction')

    parser.add_argument('--m2_aug', type=str, default='False', help='whether to augment m2 for drift readout')
    parser.add_argument('--m1_W_eig', type=str, default='False', help='project encoding onto W eigen basis')
    parser.add_argument('--m2_W_eig', type=str, default='', help='either z2x or x2z, project onto W eigen basis before decode')
    parser.add_argument('--m3_path_dep', type=str, default='', help='whether to use path dependent for m3 decoder')
    parser.add_argument('--path_dep_norm', type=str, default='False', help='whether to norm the path dependent solution for m3 decoder')
    parser.add_argument('--drift_space', type=str, default=None, help='feature, label')
    parser.add_argument('--drift_grad', type=str, default='True', help='collect gradient off drift term')
    parser.add_argument('--loss_orthog_a', type=float, default=0, help='loss orthog multiplier term')
    parser.add_argument('--householder_L', type=int, default=8, help='num iterations of householder reflection for W_orthog')
    parser.add_argument('--source_term', type=str, default='', help='describes type of source term to add')
    parser.add_argument('--q_scalar_init', type=float, default=1.0, help='[0.,1.] init of the learnable source multiplier')
    parser.add_argument('--dampen_gamma', type=float, default=1.0, help='gamma dampening coefficient, 1 is turned off, 0 is full dampening')
    parser.add_argument('--post_proc', type=str, default='none', help='post processing [none, neighbour, node]')
    parser.add_argument('--dir_grad_flow', type=str, default='none', help='directed gradient flow')

    #zinc params
    parser.add_argument('--pointwise_nonlin', type=str, default='False', help='pointwise_nonlin')
    parser.add_argument('--conv_batch_norm', type=str, default='False', help='conv_batch_norm')
    parser.add_argument('--batch', type=int, default=128, help='batch_size')
    parser.add_argument('--graph_pool', type=str, default='', help='type of graph pool operation - {add, mean}')
    parser.add_argument('--test_batches', type=int, default=None, help='reduce data size to batch num when developing')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5, help='lr_reduce_factor')
    parser.add_argument('--lr_schedule_patience', type=int, default=10, help='lr_schedule_patience')
    parser.add_argument('--lr_schedule_threshold', type=float, default=0.0001, help='lr_schedule_threshold')
    parser.add_argument('--min_lr', type=float, default=0.00001, help='min_lr')

    parser.add_argument('--lt_pointwise_nonlin', type=str, default='False', help='pointwise_nonlin')
    parser.add_argument('--lt_block_times', nargs='+', default=None, help='list of times for blocks')
    parser.add_argument('--lt_block_type', type=str, default='', help='lt_block_type')
    parser.add_argument('--lt_block_time', type=float, default=None, help='lt_block_time')
    parser.add_argument('--lt_block_step', type=float, default=None, help='lt_block_step')
    parser.add_argument('--lt_block_dimension', type=int, default=64, help='lt_block_dimension')
    parser.add_argument('--share_block', type=str, default='', help='share_block')

    args = parser.parse_args()
    opt = vars(args)

    if opt['function'] in ['greed', 'greed_scaledDP', 'greed_linear', 'greed_linear_homo', 'greed_linear_hetero',
                           'greed_non_linear', 'greed_lie_trotter', 'gcn', 'gcn2', 'mlp', 'gcn_dgl', 'gcn_res_dgl',
                           'gat', 'GAT', 'gat_dgl']:
        opt = greed_run_params(opt)  ###very basic params for GREED


    if not opt['wandb_sweep']:  # sweeps are run from YAML config so don't need these
        # this includes args for running locally - specified in YAML for tunes
        opt = not_sweep_args(opt, project_name='zinc_runs', group_name='testing')
        opt = greed_hyper_params(opt)
        opt = zinc_params(opt)

    # applied to both sweeps and not sweeps
    opt = tf_ablation_args(opt)
    main(opt)


# terminal commands for sweeps
# wandb sweep ../wandb_sweep_configs/greed_sweep_grid.yaml
# ./run_sweeps.sh XXX
# nohup ./run_sweeps.sh XXX &
# wandb sync --include-synced --include-offline --sync-all
# wandb sync --include-offline /wandb/offline-*