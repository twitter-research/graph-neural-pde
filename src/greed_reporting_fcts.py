import os
import shutil
import torch
import numpy as np
from torch_geometric.utils import degree, softmax, homophily
from torch_scatter import scatter_mean
import wandb
from torch.nn import Parameter, Softmax, Softplus
from torch.distributions import Categorical

def set_reporting_attributes(func, data, opt):
    func.get_evol_stats = False
    func.energy = 0
    
    func.attentions = []
    func.fOmf = []
    func.L2dist = []
    func.node_magnitudes = []
    func.logit_magnitudes = []
    func.node_measures = []
    func.train_accs = []
    func.val_accs = []
    func.test_accs = []
    func.homophils = []

    func.entropies = None
    func.confusions = None
    func.val_dist_mean_feat = None
    func.val_dist_sd_feat = None
    func.test_dist_mean_feat = None
    func.test_dist_sd_feat = None
    func.val_dist_mean_label = None
    func.val_dist_sd_label = None
    func.test_dist_mean_label = None
    func.test_dist_sd_label = None

    #todo what are these? where are they used?
    # graph_edge_homophily = homophily(edge_index=func.edge_index, y=data.y, method='edge')
    # graph_node_homophily = homophily(edge_index=func.edge_index, y=data.y, method='node')
    func.labels = data.y
    row, col = data.edge_index
    func.edge_homophils = torch.zeros(row.size(0), device=row.device)
    func.edge_homophils[data.y[row] == data.y[col]] = 1.
    node_homophils = scatter_mean(func.edge_homophils, col, 0, dim_size=data.y.size(0))
    #todo what are these? where are they used?
    func.node_homophils = node_homophils #in reports 4 and 5

    if opt['lie_trotter'] != 'gen_2':
      func.cum_steps_list, func.cum_time_points, func.cum_time_ticks, func.block_type_list = create_time_lists(opt)


def set_folders_pdfs(func, opt):
    savefolder = f"../plots/{opt['gnl_savefolder']}_{opt['dataset']}_{opt['gnl_W_style']}_{opt['time']}_{opt['step_size']}"
    func.savefolder = savefolder
    try:
        os.mkdir(savefolder)
    except OSError:
        if os.path.exists(savefolder):
            shutil.rmtree(savefolder)
            os.mkdir(savefolder)
            print("%s exists, clearing existing images" % savefolder)
        else:
            print("Creation of the directory %s failed" % savefolder)
    else:
        print("Successfully created the directory %s " % savefolder)


def create_time_lists(opt):
    #make lists:
    # cummaltive block_idxs for paths.npx, ie [0, ...
    # cummaltive actual times up to and including times of block
    # description of block types
    if opt['lie_trotter'] == 'gen_2':
        block_type_list = []
        for i, block_dict in enumerate(opt['lt_gen2_args']):
            block_time = block_dict['lt_block_time']
            block_step = block_dict['lt_block_step']
            block_type = block_dict['lt_block_type']
            steps = int(block_time / block_step)

            if i==0:
              cum_steps_list = [steps]
              cum_time_points = [0, block_time]
              cum_time_ticks = list(np.arange(cum_time_points[-2], cum_time_points[-1], block_step))
              block_type_list += steps * [block_type]
            else:
              cum_steps = cum_steps_list[-1] + steps
              cum_steps_list.append(cum_steps)

              block_start = cum_time_points[-1] + block_time
              cum_time_points.append(block_start)

              cum_time_ticks += list(np.arange(cum_time_points[-2], cum_time_points[-1], block_step))
              block_type_list += steps * [block_type]

        block_type_list.append(block_type)
        cum_time_ticks += [cum_time_ticks[-1] + block_step]
    else:
      block_time = opt['time']
      block_step = opt['step_size']
      steps = int(block_time / block_step)
      cum_steps_list = [steps]
      cum_time_points = [0, block_time]
      cum_time_ticks = list(np.arange(cum_time_points[-2], cum_time_points[-1], block_step)) + [block_time]
      block_type_list = steps * []
    return cum_steps_list, cum_time_points, cum_time_ticks, block_type_list

@torch.no_grad()
def test(logits, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  accs = []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs

@torch.no_grad()
def get_entropies(logits, data, activation="softmax", pos_encoding=None,
                  opt=None):  # opt required for runtime polymorphism
  entropies_dic = {}  # []
  # https://discuss.pytorch.org/t/difficulty-understanding-entropy-in-pytorch/51014
  # https://pytorch.org/docs/stable/distributions.html
  if activation == "softmax":
    S = Softmax(dim=1)
  elif activation == "squaremax":
    S = Softplus(dim=1)

  for mask_name, mask in data('train_mask', 'val_mask', 'test_mask'):
    p_matrix = S(logits[mask])
    pred = logits[mask].max(1)[1]
    labels = data.y[mask]
    correct = pred == labels
    entropy2 = Categorical(probs=p_matrix).entropy()
    entropies_dic[f"entropy_{mask_name}_correct"] = correct.unsqueeze(0)
    entropies_dic[f"entropy_{mask_name}"] = entropy2.unsqueeze(0)
  return entropies_dic

def get_confusion(func, data, pred, norm_type):
    # conf_mat = confusion_matrix(data.y, pred, normalize=norm_type)
    # torch_conf_mat = func.torch_confusion(data.y, pred, norm_type)
    # print(torch.allclose(torch.from_numpy(conf_mat), torch_conf_mat, rtol=0.001))
    # train_cm = confusion_matrix(data.y[data.train_mask], pred[data.train_mask], normalize=norm_type)
    # val_cm = confusion_matrix(data.y[data.val_mask], pred[data.val_mask], normalize=norm_type)
    # test_cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask], normalize=norm_type)
    num_class = func.C
    conf_mat = torch_confusion(func, data.y, pred, num_class, norm_type)
    train_cm = torch_confusion(func, data.y[data.train_mask], pred[data.train_mask], num_class, norm_type)
    val_cm = torch_confusion(func, data.y[data.val_mask], pred[data.val_mask], num_class, norm_type)
    test_cm = torch_confusion(func, data.y[data.test_mask], pred[data.test_mask], num_class, norm_type)
    return conf_mat, train_cm, val_cm, test_cm

def torch_confusion(func, labels, pred, num_class, norm_type):
    '''
    Truth - row i
    Pred - col j
    '''
    num_nodes = labels.shape[0]
    conf_mat = torch.zeros((num_class, num_class), dtype=torch.double, device=func.device)
    for i in range(num_class):
      for j in range(num_class):
        conf_mat[i,j] = ((labels==i).long() * (pred==j).long()).sum()
    if norm_type == None:
      pass
    elif norm_type == 'true':
      trues = torch.zeros(num_class, dtype=torch.double, device=func.device)
      for c in range(num_class):
        trues[c] = (labels == c).sum()
      conf_mat = conf_mat / trues.unsqueeze(-1)
    elif norm_type == 'pred':
      preds = torch.zeros(num_class, dtype=torch.double, device=func.device)
      for c in range(num_class):
        preds[c] = (pred == c).sum()
      conf_mat = conf_mat / preds.unsqueeze(0)
    elif norm_type == 'all':
      conf_mat / num_nodes
    return conf_mat

def get_distances(func, data, x, num_class, base_mask, eval_masks, base_type):
    #this should work for features or preds/label space
    base_av = torch.zeros((num_class, x.shape[-1]), device=func.device)
    #calculate average hidden state per class in the baseline set - [C, d]
    if base_type == 'train_avg':
      for c in range(num_class):
        base_c_mask = data.y[base_mask] == c
        base_av_c = x[base_mask][base_c_mask].mean(dim=0)
        base_av[c] = base_av_c
    elif base_type == 'e_k':
      base_av = torch.eye(num_class, device=func.device)

    #for every node calcualte the L2 distance - [N, C] and [N, C]
    #todo for label space calc distance from Ek
    dist = x.unsqueeze(-1) - base_av.T.unsqueeze(0)
    L2_dist = torch.sqrt(torch.sum(dist**2, dim=1))

    #for every node in each true class in the val/test sets calc the distances away from the average train set for each class
    eval_means = []
    eval_sds = []
    for eval_mask in eval_masks:
      eval_dist_mean = torch.zeros((num_class, num_class), device=func.device)
      eval_dist_sd = torch.zeros((num_class, num_class), device=func.device)
      for c in range(num_class):
        base_c_mask = data.y[eval_mask] == c
        eval_dist_mean[c] = L2_dist[eval_mask][base_c_mask].mean(dim=0)
        eval_dist_sd[c] = L2_dist[eval_mask][base_c_mask].std(dim=0)

      eval_means.append(eval_dist_mean)
      eval_sds.append(eval_dist_sd)
    #output: rows base_class, cols eval_class
    return eval_means, eval_sds


def generate_stats(func, t, x, f):
    # get edge stats if not a diffusion pass/block
    # if func.do_drift(t):
    src_x, dst_x = func.get_src_dst(x)
    fOmf, attention = func.calc_dot_prod_attention(src_x, dst_x)

    if func.opt['lie_trotter'] == 'gen_2' and not func.do_drift(t):
        if func.opt['lt_block_type'] == 'threshold':
            src_x, dst_x = func.get_src_dst(x)
            fOmf, attention = func.calc_dot_prod_attention(src_x, dst_x)

    # todo these energy formulas are wrong
    if func.opt['gnl_style'] == 'scaled_dot':
        if func.opt['gnl_activation'] == "sigmoid_deriv":
            energy = torch.sum(torch.sigmoid(fOmf))
        elif func.opt['gnl_activation'] == "squareplus_deriv":
            energy = torch.sum((fOmf + torch.sqrt(fOmf ** 2 + 4)) / 2)
        elif func.opt['gnl_activation'] == "exponential":
            energy = torch.sum(torch.exp(fOmf))
        elif func.opt['gnl_activation'] == "identity":
            energy = fOmf ** 2 / 2
    else:
        energy = 0
        energy = energy + 0.5 * func.delta * torch.sum(x ** 2)

        if func.opt['test_mu_0'] and func.opt['add_source']:
            energy = energy - func.beta_train * torch.sum(x * func.x0)
        elif not func.opt['test_mu_0']:
            energy = energy + func.mu * torch.sum((x - func.x0) ** 2)
        else:
            energy = 0
            func.energy = energy

        wandb.log({f"gf_e{func.epoch}_energy_change": energy - func.energy, f"gf_e{func.epoch}_energy": energy,
                   f"gf_e{func.epoch}_f": (f ** 2).sum(),
                   f"gf_e{func.epoch}_x": (x ** 2).sum(),
                   "grad_flow_step": func.wandb_step})

        if func.opt['lie_trotter'] == 'gen_2':
            if func.opt['lt_block_type'] == 'label':
                logits = x
                pred = logits.max(1)[1]
            else:
                logits, pred = func.predict(x)
        else:
            logits, pred = func.predict(x)
        sm_logits = torch.softmax(logits, dim=1)
        train_acc, val_acc, test_acc = test(logits, func.data)
        homophil = homophily(edge_index=func.edge_index, y=pred)
        L2dist = torch.sqrt(torch.sum((src_x - dst_x) ** 2, dim=1))
        conf_mat, train_cm, val_cm, test_cm = get_confusion(func, func.data, pred, norm_type='true')  # 'all')
        eval_means_feat, eval_sds_feat = get_distances(func, func.data, x, func.C, base_mask=func.data.train_mask,
                                                            eval_masks=[func.data.val_mask, func.data.test_mask],
                                                            base_type="train_avg")
        # eval_means_label, eval_sds_label = get_distances(func, func.data, logits, func.C, base_mask=func.data.train_mask,
        #                                                       eval_masks=[func.data.val_mask, func.data.test_mask],
        #                                                       base_type="train_avg")
        # eval_means_label, eval_sds_label = get_distances(func, func.data, sm_logits, func.C, base_mask=func.data.train_mask,
        #                                                       eval_masks=[func.data.val_mask, func.data.test_mask],
        #                                                       base_type="e_k")
        eval_means_label, eval_sds_label = get_distances(func, func.data, logits, func.C, base_mask=func.data.train_mask,
                                                              eval_masks=[func.data.val_mask, func.data.test_mask],
                                                              base_type="e_k")

        entropies = get_entropies(logits, func.data)

    return fOmf, logits, attention, L2dist, train_acc, val_acc, test_acc, homophil, conf_mat, train_cm, val_cm, test_cm,\
           eval_means_feat, eval_sds_feat, eval_means_label, eval_sds_label, entropies


# todo this could be easily condensed/optimised if just saved all objects in lists and then do the stacking at the end
# above job is now half done apart from entropies, confusions and distances
def append_stats(func, attention, fOmf, logits, x, measure, L2dist, train_acc, val_acc, test_acc, homophil, conf_mat, train_cm, val_cm, test_cm,
           eval_means_feat, eval_sds_feat, eval_means_label, eval_sds_label, entropies):

    func.attentions.append(attention)
    func.fOmf.append(fOmf)
    func.L2dist.append(L2dist)
    func.node_magnitudes.append(torch.sqrt(torch.sum(x ** 2, dim=1)))
    func.logit_magnitudes.append(torch.sqrt(torch.sum(logits ** 2, dim=1)))
    func.node_measures.append(measure.detach())
    func.train_accs.append(train_acc)
    func.val_accs.append(val_acc)
    func.test_accs.append(test_acc)
    func.homophils.append(homophil)

    if len(func.attentions) is 1:
        func.entropies = entropies
        func.confusions = [conf_mat, train_cm, val_cm, test_cm]

        func.val_dist_mean_feat = eval_means_feat[0]
        func.val_dist_sd_feat = eval_sds_feat[0]
        func.test_dist_mean_feat = eval_means_feat[1]
        func.test_dist_sd_feat = eval_sds_feat[1]

        func.val_dist_mean_label = eval_means_label[0]
        func.val_dist_sd_label = eval_sds_label[0]
        func.test_dist_mean_label = eval_means_label[1]
        func.test_dist_sd_label = eval_sds_label[1]
    else:
        temp_entropies = entropies
        for key, value, in func.entropies.items():
            func.entropies[key] = torch.cat([value, temp_entropies[key]], dim=0)

        if len(func.confusions[0].shape) == 2:
            func.confusions[0] = torch.stack((func.confusions[0], conf_mat), dim=-1)
            func.confusions[1] = torch.stack((func.confusions[1], train_cm), dim=-1)
            func.confusions[2] = torch.stack((func.confusions[2], val_cm), dim=-1)
            func.confusions[3] = torch.stack((func.confusions[3], test_cm), dim=-1)

            func.val_dist_mean_feat = torch.stack((func.val_dist_mean_feat, eval_means_feat[0]), dim=-1)
            func.val_dist_sd_feat = torch.stack((func.val_dist_sd_feat, eval_sds_feat[0]), dim=-1)
            func.test_dist_mean_feat = torch.stack((func.test_dist_mean_feat, eval_means_feat[1]), dim=-1)
            func.test_dist_sd_feat = torch.stack((func.test_dist_sd_feat, eval_sds_feat[1]), dim=-1)

            func.val_dist_mean_label = torch.stack((func.val_dist_mean_label, eval_means_label[0]), dim=-1)
            func.val_dist_sd_label = torch.stack((func.val_dist_sd_label, eval_sds_label[0]), dim=-1)
            func.test_dist_mean_label = torch.stack((func.test_dist_mean_label, eval_means_label[1]), dim=-1)
            func.test_dist_sd_label = torch.stack((func.test_dist_sd_label, eval_sds_label[1]), dim=-1)
        else:
            func.confusions[0] = torch.cat((func.confusions[0], conf_mat.unsqueeze(-1)), dim=-1)
            func.confusions[1] = torch.cat((func.confusions[1], train_cm.unsqueeze(-1)), dim=-1)
            func.confusions[2] = torch.cat((func.confusions[2], val_cm.unsqueeze(-1)), dim=-1)
            func.confusions[3] = torch.cat((func.confusions[3], test_cm.unsqueeze(-1)), dim=-1)

            func.val_dist_mean_feat = torch.cat((func.val_dist_mean_feat, eval_means_feat[0].unsqueeze(-1)), dim=-1)
            func.val_dist_sd_feat = torch.cat((func.val_dist_sd_feat, eval_sds_feat[0].unsqueeze(-1)), dim=-1)
            func.test_dist_mean_feat = torch.cat((func.test_dist_mean_feat, eval_means_feat[1].unsqueeze(-1)), dim=-1)
            func.test_dist_sd_feat = torch.cat((func.test_dist_sd_feat, eval_sds_feat[1].unsqueeze(-1)), dim=-1)

            func.val_dist_mean_label = torch.cat((func.val_dist_mean_label, eval_means_label[0].unsqueeze(-1)), dim=-1)
            func.val_dist_sd_label = torch.cat((func.val_dist_sd_label, eval_sds_label[0].unsqueeze(-1)), dim=-1)
            func.test_dist_mean_label = torch.cat((func.test_dist_mean_label, eval_means_label[1].unsqueeze(-1)), dim=-1)
            func.test_dist_sd_label = torch.cat((func.test_dist_sd_label, eval_sds_label[1].unsqueeze(-1)), dim=-1)

def stack_stats(func):
    func.attentions = torch.stack(func.attentions)
    func.fOmf = torch.stack(func.fOmf)
    func.L2dist = torch.stack(func.L2dist)
    func.node_magnitudes = torch.stack(func.node_magnitudes)
    func.logit_magnitudes = torch.stack(func.logit_magnitudes)
    func.node_measures = torch.stack(func.node_measures)
    # func.train_accs = torch.stack(func.train_accs) #these are lists of floats
    # func.val_accs = torch.stack(func.val_accs)
    # func.test_accs = torch.stack(func.test_accs)
    # func.homophils = torch.stack(func.homophils)


def reset_stats(func):
    func.attentions = []
    func.fOmf = []
    func.L2dist = []
    func.node_magnitudes = []
    func.logit_magnitudes = []
    func.node_measures = []
    func.train_accs = []
    func.val_accs = []
    func.test_accs = []
    func.homophils = []

    func.entropies = None
    func.confusions = None
    func.val_dist_mean_feat = None
    func.val_dist_sd_feat = None
    func.test_dist_mean_feat = None
    func.test_dist_sd_feat = None
    func.val_dist_mean_label = None
    func.val_dist_sd_label = None
    func.test_dist_mean_label = None
    func.test_dist_sd_label = None

