import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import to_networkx
import wandb
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import check_random_state

from reports_TSNE import tsne_snap, tsne_ani, tsne_full, project_paths_label_space

def report_1(ax, fig, odefunc, row, epoch):
    '''spectrum'''
    opt = odefunc.opt
    ###1) multi grid Omega spectrum charts
    # spectral and accuracy plots
    if opt['gnl_style'] == 'scaled_dot':
        Omega = odefunc.Omega
    elif opt['gnl_style'] == 'general_graph':
        Omega = odefunc.gnl_W

    L, Q = torch.linalg.eigh(Omega)  # fast version for symmetric matrices https://pytorch.org/docs/stable/generated/torch.linalg.eig.html

    mat = ax[row, 0].matshow(Omega.cpu().numpy(), interpolation='nearest')
    ax[row, 0].xaxis.set_tick_params(labelsize=16)
    ax[row, 0].yaxis.set_tick_params(labelsize=16)
    cbar = fig.colorbar(mat, ax=ax[row, 0], shrink=0.75)
    cbar.ax.tick_params(labelsize=16)

    ax[row, 1].bar(range(L.shape[0]), L.cpu().numpy(), width=1.0, zorder=0)
    if opt['gnl_W_style'] == 'diag_dom':
        sort_ta = torch.sort(odefunc.t_a)[0]
        sort_ra = torch.sort(odefunc.r_a)[0]
        ax[row, 1].plot(range(L.shape[0]), sort_ta.cpu().numpy(), c='tab:green', label='t_a', zorder=10)
        ax[row, 1].plot(range(L.shape[0]), sort_ra.cpu().numpy(), c='tab:orange', label='r_a', zorder=5)
        ax[row, 1].legend()

    # ax[row, 1].set_title(f"W, Eig-vals, Eig-vecs, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    ax[row, 1].set_title(f"{odefunc.opt['dataset']}, W: Eig-vals, Eig-vecs, epoch {epoch}", fontdict={'fontsize': 24})
    ax[row, 1].xaxis.set_tick_params(labelsize=16)
    ax[row, 1].yaxis.set_tick_params(labelsize=16)

    mat2 = ax[row, 2].matshow(Q.cpu().numpy(), interpolation='nearest')
    ax[row, 2].xaxis.set_tick_params(labelsize=16)
    ax[row, 2].yaxis.set_tick_params(labelsize=16)
    cbar1 = fig.colorbar(mat2, ax=ax[row, 2], shrink=0.75)
    cbar1.ax.tick_params(labelsize=16)
    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
        fig.show()


def report_2(ax, fig, odefunc, row, epoch):
    '''acc_entropy'''
    opt = odefunc.opt
    train_accs = odefunc.train_accs
    val_accs = odefunc.val_accs
    test_accs = odefunc.test_accs
    homophils = odefunc.homophils

    ax[row, 0].plot(np.arange(0.0, len(train_accs) * opt['step_size'], opt['step_size']), train_accs,
                                label="train")
    ax[row, 0].plot(np.arange(0.0, len(val_accs) * opt['step_size'], opt['step_size']), val_accs,
                                label="val")
    ax[row, 0].plot(np.arange(0.0, len(test_accs) * opt['step_size'], opt['step_size']), test_accs,
                                label="test")
    ax[row, 0].plot(np.arange(0.0, len(homophils) * opt['step_size'], opt['step_size']), homophils,
                                label="homophil")
    ax[row, 0].xaxis.set_tick_params(labelsize=16)
    ax[row, 0].yaxis.set_tick_params(labelsize=16)
    ax[row, 0].set_title(f"Accuracy evolution, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    ax[row, 0].legend(loc="upper right", fontsize=24)
    # fig.show()

    # entropy plots #getting the line colour to change
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
    # https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection
    entropies = odefunc.entropies

    # fig, ax = plt.subplots() #figsize=(8, 16))
    x = np.arange(0.0, entropies['entropy_train_mask'].shape[0] * opt['step_size'], opt['step_size'])
    ys = entropies['entropy_train_mask'].cpu().numpy()
    # ax.set_xlim(np.min(x), np.max(x))
    # ax.set_ylim(np.min(ys), np.max(ys))
    ax[row, 1].set_xlim(np.min(x), np.max(x))
    ax[row, 1].set_ylim(np.min(ys), np.max(ys))
    ax[row, 1].xaxis.set_tick_params(labelsize=16)
    ax[row, 1].yaxis.set_tick_params(labelsize=16)

    cmap = ListedColormap(['r', 'g'])
    norm = BoundaryNorm([-1, 0.5, 2.0], cmap.N)
    for i in range(entropies['entropy_train_mask'].shape[1]):
        tf = entropies['entropy_train_mask_correct'][:, i].float().cpu().numpy()
        points = np.expand_dims(np.concatenate([x.reshape(-1, 1),
                                                entropies['entropy_train_mask'][:, i].reshape(-1, 1).cpu().numpy()], axis=1), axis=1)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm)
        # lc.set_array(tf[:-1]) #correctness at time at start of segment decides colouring
        lc.set_array(tf[1:]) #correctness at time at end of segment decides colouring
        # ax.add_collection(lc)
        ax[row, 1].add_collection(lc)

    # ax.set_title(f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}")
    ax[row, 1].set_title(
        f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}, block {odefunc.block_num}",
        fontdict={'fontsize': 24})
    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
        fig.show()

def report_3(ax, fig, odefunc, row, epoch):
    '''edge_evol'''
    opt = odefunc.opt
    fOmf = odefunc.fOmf
    edge_homophils = odefunc.edge_homophils
    ax[row, 0].plot(np.arange(0.0, fOmf.shape[0] * opt['step_size'], opt['step_size']), fOmf.cpu().numpy())
    # too slow potentially can use pandas for the color map
    # colormap = cm.Spectral  # jet
    # normalize = mcolors.Normalize(vmin=edge_homophils.min(), vmax=edge_homophils.max())
    # for e in range(fOmf.shape[1]):
    #   color = colormap(normalize(edge_homophils[e]))
    #   ax[row,0].plot(np.arange(0.0, fOmf.shape[0] * opt['step_size'], opt['step_size']),
    #                          fOmf, c=color)
    ax[row, 0].xaxis.set_tick_params(labelsize=16)
    ax[row, 0].yaxis.set_tick_params(labelsize=16)
    ax[row, 0].set_title(f"fOmf, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})

    L2dist = odefunc.L2dist
    ax[row, 1].plot(np.arange(0.0, L2dist.shape[0] * opt['step_size'], opt['step_size']), L2dist.cpu().numpy())
    ax[row, 1].xaxis.set_tick_params(labelsize=16)
    ax[row, 1].yaxis.set_tick_params(labelsize=16)
    ax[row, 1].set_title(f"L2dist, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
        fig.show()

def report_4(ax, fig, odefunc, row, epoch):
    '''node_evol'''
    opt = odefunc.opt
    #node magnitudes
    magnitudes = odefunc.node_magnitudes
    ax[row, 0].plot(np.arange(0.0, magnitudes.shape[0] * opt['step_size'], opt['step_size']), magnitudes.cpu().numpy())
    # labels  = odefunc.labels
    # colormap = cm.Spectral  # jet
    # normalize = mcolors.Normalize(vmin=labels.min(), vmax=labels.max())
    # for n in range(magnitudes.shape[1]):
    #   color = colormap(normalize(labels[n]))
    #   ax[row, 0].plot(np.arange(0.0, magnitudes.shape[0] * opt['step_size'], opt['step_size']),
    #                             magnitudes[:,n], color=color)
    ax[row, 0].xaxis.set_tick_params(labelsize=16)
    ax[row, 0].yaxis.set_tick_params(labelsize=16)
    ax[row, 0].set_title(f"f magnitudes, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})

    measures = odefunc.node_measures
    ax[row, 1].plot(np.arange(0.0, measures.shape[0] * opt['step_size'], opt['step_size']), measures.cpu().numpy())
    node_homophils = odefunc.node_homophils
    # for n in range(measures.shape[1]):
    #   normalize = mcolors.Normalize(vmin=node_homophils.min(), vmax=node_homophils.max())
    #   colormap = cm.Spectral #jet
    #   color = colormap(normalize(node_homophils[n]))
    #   ax[row, 1].plot(np.arange(0.0, measures.shape[0] * opt['step_size'], opt['step_size']),
    #                             measures[:,n], color=color)
    ax[row, 1].xaxis.set_tick_params(labelsize=16)
    ax[row, 1].yaxis.set_tick_params(labelsize=16)
    ax[row, 1].set_title(f"Node measures, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    # fig.show()

    confusions = odefunc.confusions
    colormap = cm.get_cmap(name="Set1")
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    conf_set = ['all','train', 'val', 'test']
    for i, conf_mat in enumerate(confusions):
        for c in range(conf_mat.shape[0]):
            correct = conf_mat[c,c,:]
            ax[row, 2].plot(np.arange(0.0, correct.shape[0] * opt['step_size'], opt['step_size']),
                                  correct.cpu().numpy(), color=colormap(c), linestyle=linestyles[i], label=f"{conf_set[i]}_c{c}")
            # incorrect = torch.sum(conf_mat[c], dim=0) - correct #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html - sum over predictions j=cols
            # ax[row, 1].plot(np.arange(0.0, incorrect.shape[0] * opt['step_size'], opt['step_size']),
            #                       incorrect, color=colormap(c), linestyle=linestyles[i], label=f"{conf_set[i]}_c{c}")

    ax[row, 2].xaxis.set_tick_params(labelsize=16)
    ax[row, 2].yaxis.set_tick_params(labelsize=16)
    ax[row, 2].set_title(f"Correct preds evol, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    ax[row, 2].legend()
    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
        fig.show()

def report_5(ax, fig, odefunc, row, epoch):
    '''node_scatter'''
    opt = odefunc.opt
    # node magnitude against degree or homophilly, colour is class
    magnitudes = odefunc.node_magnitudes
    node_homophils = odefunc.node_homophils
    labels = odefunc.labels
    ax[row, 0].scatter(x=magnitudes[-1, :].cpu().numpy(), y=node_homophils.cpu().numpy(), c=labels.cpu().numpy())  # , cmap='Set1')
    ax[row, 0].xaxis.set_tick_params(labelsize=16)
    ax[row, 0].yaxis.set_tick_params(labelsize=16)
    ax[row, 0].set_title(f"f magnitudes v node homophils, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})

    # node measure against degree or homophilly, colour is class
    measures = odefunc.node_measures
    ax[row, 1].scatter(x=measures[-1, :].cpu().numpy(), y=node_homophils.cpu().numpy(), c=labels.cpu().numpy())  # , cmap='Set1')
    ax[row, 1].xaxis.set_tick_params(labelsize=16)
    ax[row, 1].yaxis.set_tick_params(labelsize=16)
    ax[row, 1].set_title(f"Node measures v node homophils, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
        fig.show()

def report_6(ax, fig, odefunc, row, epoch):
    '''edge_scatter'''
    opt = odefunc.opt
    # edge dot product against edge distance, coloured by edge homopholliy
    fOmf = odefunc.fOmf
    L2dist = odefunc.L2dist
    edge_homophils = odefunc.edge_homophils
    mask = edge_homophils == 1
    ax[row].scatter(x=fOmf[-1, :][mask].cpu().numpy(), y=L2dist[-1, :][mask].cpu().numpy(), label='edge:1',
                                 c='gold')  # c=edge_homophils[mask])
    ax[row].scatter(x=fOmf[-1, :][~mask].cpu().numpy(), y=L2dist[-1, :][~mask].cpu().numpy(), label='edge:0',
                                 c='indigo')  # c=edge_homophils[~mask])

    ax[row].xaxis.set_tick_params(labelsize=16)
    ax[row].yaxis.set_tick_params(labelsize=16)
    ax[row].set_title(f"Edge fOmf against L2, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    ax[row].legend(loc="upper right", fontsize=24)
    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
        fig.show()

def report_7(ax, fig, odefunc, row, epoch):
    '''class_dist'''
    opt = odefunc.opt
    #column 0
    val_dist_mean_feat = odefunc.val_dist_mean_feat
    val_dist_sd_feat = odefunc.val_dist_sd_feat
    test_dist_mean_feat = odefunc.test_dist_mean_feat
    test_dist_sd_feat = odefunc.test_dist_sd_feat
    colormap = cm.get_cmap(name="Set1")
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for c in range(val_dist_mean_feat.shape[0]):
        # plot diags
        ax[row, 0].plot(np.arange(0.0, val_dist_mean_feat.shape[-1] * opt['step_size'], opt['step_size']),
                                  val_dist_mean_feat[c,c,:].cpu().numpy(), color=colormap(c), linestyle=linestyles[0],
                                  label=f"base_{c}_eval_{c}")
        # output: rows base_class, cols eval_class
        for i in range(val_dist_mean_feat.shape[0]):
            ax[row, 0].plot(np.arange(0.0, val_dist_mean_feat.shape[-1] * opt['step_size'], opt['step_size']),
                                       val_dist_mean_feat[i, c, :].cpu().numpy(), color=colormap(c),
                                       linestyle=linestyles[1],
                                       label=f"base_non{c}_eval_{c}")
    ax[row, 0].xaxis.set_tick_params(labelsize=16)
    ax[row, 0].yaxis.set_tick_params(labelsize=16)
    ax[row, 0].set_title(f"Class feature L2 distances evol, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    ax[row, 0].legend()

    #column 1
    val_dist_mean_label = odefunc.val_dist_mean_label
    for c in range(val_dist_mean_label.shape[0]):
        # plot diags
        ax[row, 1].plot(np.arange(0.0, val_dist_mean_label.shape[-1] * opt['step_size'], opt['step_size']),
                                  val_dist_mean_label[c,c,:].cpu().numpy(), color=colormap(c), linestyle=linestyles[0],
                                  label=f"base_{c}_eval_{c}")
        # output: rows base_class, cols eval_class
        for i in range(val_dist_mean_label.shape[0]):
            ax[row, 1].plot(np.arange(0.0, val_dist_mean_label.shape[-1] * opt['step_size'], opt['step_size']),
                                       val_dist_mean_label[i, c, :].cpu().numpy(), color=colormap(c),
                                       linestyle=linestyles[1],
                                       label=f"base_non{c}_eval_{c}")
    ax[row, 1].xaxis.set_tick_params(labelsize=16)
    ax[row, 1].yaxis.set_tick_params(labelsize=16)
    ax[row, 1].set_title(f"Class label L2 distances evol, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    ax[row, 1].legend()

    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
        fig.show()


def save_TSNE_stats(odefunc, epoch):
    # savefolder = f"../plots/{odefunc.opt['gnl_savefolder']}_{odefunc.opt['dataset']}"
    savefolder = odefunc.savefolder
    npy_path = savefolder + f"/paths_epoch{epoch}_{odefunc.opt['dataset']}.npy"
    npy_label = savefolder + f"/labels_epoch{epoch}_{odefunc.opt['dataset']}.npy"
    npy_path_stack = torch.stack(odefunc.paths, dim=-1).detach().cpu().numpy()
    np.save(npy_path, npy_path_stack) #np.stack(odefunc.paths, axis=-1))
    np.save(npy_label, odefunc.labels.detach().cpu().numpy())
    G = to_networkx(odefunc.data)
    nx.write_gpickle(G, savefolder + f"/nxgraph_epoch{epoch}_{odefunc.opt['dataset']}.pkl")
    #save opt
    with open(savefolder + '/opt.json', 'w+') as f:
        try:
            json.dump(odefunc.opt._as_dict(), f, indent=4)
        except:
            json.dump(odefunc.opt, f, indent=4)
    #save decoder - https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
    m2 = odefunc.GNN_m2
    torch.save(m2, savefolder + f"/m2_epoch{epoch}_{odefunc.opt['dataset']}.pt")


def report_8(ax, fig, odefunc, row, epoch):
    '''evol TSNE'''
    #     tsne_snap(ax, fig, odefunc, row, epoch, savefolder)
    save_TSNE_stats(odefunc, epoch)
    tsne_snap(ax, fig, odefunc, row, epoch)

    #doesn't do anything currently
    if epoch == odefunc.opt['wandb_epoch_list'][-1]:
        X = torch.stack(odefunc.paths, dim=-1) #np.stack(odefunc.paths, axis=-1)
        labels = np.stack(odefunc.labels, axis=-1)
        G = to_networkx(odefunc.data)  # for optional including of the graph
        m2 = odefunc.GNN_m2
        X = project_paths_label_space(m2, X)
        centers = np.eye(odefunc.C)[..., np.newaxis].repeat(repeats=X.shape[-1], axis=-1)

        # https://docs.wandb.ai/guides/artifacts
        # artifact = wandb.Artifact('model', type='model')
        # artifact.add_file('save/to/path/model.pth')
        # run.log_artifact(artifact)


def report_9(ax, fig, odefunc, row, epoch):
    '''val and test set entropies'''
    opt = odefunc.opt
    # entropy plots #getting the line colour to change
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
    # https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection
    entropies = odefunc.entropies

    def do_plot(col, tvt_set):
        # fig, ax = plt.subplots() #figsize=(8, 16))
        x = np.arange(0.0, entropies[f'entropy_{tvt_set}_mask'].shape[0] * opt['step_size'], opt['step_size'])
        ys = entropies[f'entropy_{tvt_set}_mask'].cpu().numpy()
        # ax.set_xlim(np.min(x), np.max(x))
        # ax.set_ylim(np.min(ys), np.max(ys))
        ax[row, col].set_xlim(np.min(x), np.max(x))
        ax[row, col].set_ylim(np.min(ys), np.max(ys))
        ax[row, col].xaxis.set_tick_params(labelsize=16)
        ax[row, col].yaxis.set_tick_params(labelsize=16)

        cmap = ListedColormap(['r', 'g'])
        norm = BoundaryNorm([-1, 0.5, 2.0], cmap.N)
        for i in range(entropies[f'entropy_{tvt_set}_mask'].shape[1]):
            tf = entropies[f'entropy_{tvt_set}_mask_correct'][:, i].float().cpu().numpy()
            points = np.expand_dims(np.concatenate([x.reshape(-1, 1),
                                                    entropies[f'entropy_{tvt_set}_mask'][:, i].reshape(-1, 1).cpu().numpy()], axis=1), axis=1)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm)
            # lc.set_array(tf[:-1]) #correctness at time at start of segment decides colouring
            lc.set_array(tf[1:]) #correctness at time at end of segment decides colouring
            # ax.add_collection(lc)
            ax[row, col].add_collection(lc)

        # ax.set_title(f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}")
        ax[row, col].set_title(
            f"{tvt_set.title()} set (num_nodes {entropies[f'entropy_{tvt_set}_mask'].shape[1]}) Entropy, epoch {epoch}, block {odefunc.block_num}",
            fontdict={'fontsize': 24})
    plots = [[0, 'val'],[1, 'test']]
    for col, tvt_set in plots:
        do_plot(col, tvt_set)

    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
        fig.show()


def run_reports(odefunc):
    opt = odefunc.opt
    epoch = odefunc.epoch

    # find position of current epoch in epoch list
    idx = opt['wandb_epoch_list'].index(epoch)
    # determine index % num per page
    num_rows = 4
    row = idx % num_rows

    # self.pdf_list = ['spectrum', 'acc_entropy', 'edge_evol', 'node_evol', 'node_scatter', 'edge_scatter']
    #desc, cols, widths, fig_size
    opt['fig_dims'] = {1: ['spectrum', 3, [1, 1, 1], (24, 32)],
                2: ['acc_entropy', 2, [1, 1], (24, 32)],
                3: ['edge_evol', 2, [1, 1], (24, 32)],
                4: ['node_evol', 3, [1, 1, 2], (24, 32)],
                5: ['node_scatter', 2, [1, 1], (24, 32)],
                6: ['edge_scatter', 1, [1], (24, 32)],
                7: ['class_dist', 2, [1, 1], (24, 32)],
                8: ['tsne_evol', 4, [1, 1, 1, 1], (32,32)],
                9: ['v_t_entropies', 2, [1, 1], (24, 32)]}

    reports_nums = opt['reports_list']
    for rep_num in reports_nums:
        print(f"epoch {epoch}, report {rep_num}")
        report_func = globals()[f"report_{rep_num}"]

        if row == 0:  # create new figs and pdfs
            desc, cols, widths, fig_size = odefunc.opt['fig_dims'][rep_num]
            fig, ax = plt.subplots(num_rows, cols, gridspec_kw={'width_ratios': widths}, figsize=fig_size)
            if hasattr(odefunc, f"report{str(rep_num)}_fig_list"):
                getattr(odefunc, f"report{str(rep_num)}_fig_list").append([fig, ax])
            else:
                setattr(odefunc, f"report{str(rep_num)}_fig_list", [[fig, ax]])
                # savefolder = f"../plots/{opt['gnl_savefolder']}"
                # savefolder = f"../plots/{opt['gnl_savefolder']}_{opt['dataset']}"
                savefolder = odefunc.savefolder
                setattr(odefunc, f"report{str(rep_num)}_pdf", PdfPages(f"{savefolder}/{desc}_{odefunc.opt['dataset']}.pdf"))
        else:
            fig, ax = getattr(odefunc, f"report{str(rep_num)}_fig_list")[-1]

        report_func(ax, fig, odefunc, row, epoch)

        #save fig to pdf
        if (row == num_rows - 1 or epoch == opt['wandb_epoch_list'][-1]) and opt['save_local_reports']:
            getattr(odefunc, f"report{str(rep_num)}_pdf").savefig(fig)
        #log fig to wandb
        if (row == num_rows - 1 or epoch == opt['wandb_epoch_list'][-1]) and opt['save_wandb_reports']:
            wandb.log({f"report{str(rep_num)}_fig_{idx // num_rows}": wandb.Image(fig)})
        #close pdf
        if epoch == opt['wandb_epoch_list'][-1] and opt['save_local_reports']:
            getattr(odefunc, f"report{str(rep_num)}_pdf").close()


def reset_stats(odefunc):
    odefunc.fOmf = None
    odefunc.attentions = None
    odefunc.L2dist = None
    odefunc.node_magnitudes = None
    odefunc.node_measures = None
    odefunc.train_accs = None
    odefunc.val_accs = None
    odefunc.test_accs = None
    odefunc.entropies = None

    odefunc.val_dist_mean_feat = None
    odefunc.val_dist_sd_feat = None
    odefunc.test_dist_mean_feat = None
    odefunc.test_dist_sd_feat = None
    odefunc.val_dist_mean_label = None
    odefunc.val_dist_sd_label = None
    odefunc.test_dist_mean_label = None
    odefunc.test_dist_sd_label = None
    odefunc.paths = []

@torch.no_grad()
def reports_manager(model, data):
    # forward pass through the model in eval mode to generate the data
    model.odeblock.odefunc.get_evol_stats = True
    pred = model.forward(data.x).max(1)[1]
    model.odeblock.odefunc.get_evol_stats = False

    if model.opt['lie_trotter'] == 'gen_2':
        for func in model.odeblock.funcs:
            func.epoch = model.odeblock.odefunc.epoch
            func.get_evol_stats = model.odeblock.odefunc.get_evol_stats
            func.wandb_step = model.odeblock.odefunc.wandb_step
            run_reports(func)
        for func in model.odeblock.funcs:
            reset_stats(func)
    else:
        func = model.odeblock.odefunc
        func.block_num = 0
        run_reports(func)
        reset_stats(func)
