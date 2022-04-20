import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import wandb
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm


def run_reports(epoch, model, data, opt):
    # find position of current epoch in epoch list
    idx = opt['wandb_epoch_list'].index(epoch)
    # determine index % num per page
    num_rows = 4
    row = idx % num_rows
    if row == 0:  # create new figs
        spectrum_fig, spectrum_ax = plt.subplots(num_rows, 3, gridspec_kw={'width_ratios': [1, 1, 1]}, figsize=(24, 32))
        model.odeblock.odefunc.spectrum_fig_list.append([spectrum_fig, spectrum_ax])

        acc_entropy_fig, acc_entropy_ax = plt.subplots(num_rows, 2, gridspec_kw={'width_ratios': [1, 1]},
                                                       figsize=(24, 32))
        model.odeblock.odefunc.acc_entropy_fig_list.append([acc_entropy_fig, acc_entropy_ax])

        edge_evol_fig, edge_evol_ax = plt.subplots(num_rows, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(24, 32))
        model.odeblock.odefunc.edge_evol_fig_list.append([edge_evol_fig, edge_evol_ax])

        node_evol_fig, node_evol_ax = plt.subplots(num_rows, 3, gridspec_kw={'width_ratios': [1, 1, 2]}, figsize=(24, 32))
        model.odeblock.odefunc.node_evol_fig_list.append([node_evol_fig, node_evol_ax])

        node_scatter_fig, node_scatter_ax = plt.subplots(num_rows, 2, gridspec_kw={'width_ratios': [1, 1]},
                                                         figsize=(24, 32))
        model.odeblock.odefunc.node_scatter_fig_list.append([node_scatter_fig, node_scatter_ax])

        edge_scatter_fig, edge_scatter_ax = plt.subplots(num_rows, 1, gridspec_kw={'width_ratios': [1]},
                                                         figsize=(24, 32))
        model.odeblock.odefunc.edge_scatter_fig_list.append([edge_scatter_fig, edge_scatter_ax])

        class_dist_fig, class_dist_ax = plt.subplots(num_rows, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(24, 32))
        model.odeblock.odefunc.class_dist_fig_list.append([class_dist_fig, class_dist_ax])
    else:
        spectrum_fig, spectrum_ax = model.odeblock.odefunc.spectrum_fig_list[-1]
        acc_entropy_fig, acc_entropy_ax = model.odeblock.odefunc.acc_entropy_fig_list[-1]
        edge_evol_fig, edge_evol_ax = model.odeblock.odefunc.edge_evol_fig_list[-1]
        node_evol_fig, node_evol_ax = model.odeblock.odefunc.node_evol_fig_list[-1]
        node_scatter_fig, node_scatter_ax = model.odeblock.odefunc.node_scatter_fig_list[-1]
        edge_scatter_fig, edge_scatter_ax = model.odeblock.odefunc.edge_scatter_fig_list[-1]
        class_dist_fig, class_dist_ax = model.odeblock.odefunc.class_dist_fig_list[-1]

    # forward pass through the model in eval mode to generate the data
    model.odeblock.odefunc.get_evol_stats = True
    pred = model.forward(data.x).max(1)[1]
    model.odeblock.odefunc.get_evol_stats = False

    # spectral and accuracy plots
    if opt['gnl_style'] == 'scaled_dot':
        Omega = model.odeblock.odefunc.Omega
    elif opt['gnl_style'] == 'general_graph':
        Omega = model.odeblock.odefunc.gnl_W

    L, Q = torch.linalg.eigh(
        Omega)  # fast version for symmetric matrices https://pytorch.org/docs/stable/generated/torch.linalg.eig.html

    ###1) multi grid Omega spectrum charts
    mat = spectrum_ax[row, 0].matshow(Omega.cpu().numpy(), interpolation='nearest')
    spectrum_ax[row, 0].xaxis.set_tick_params(labelsize=16)
    spectrum_ax[row, 0].yaxis.set_tick_params(labelsize=16)
    cbar = spectrum_fig.colorbar(mat, ax=spectrum_ax[row, 0], shrink=0.75)
    cbar.ax.tick_params(labelsize=16)

    spectrum_ax[row, 1].bar(range(L.shape[0]), L.cpu().numpy(), width=1.0, zorder=0)
    if opt['gnl_W_style'] == 'diag_dom':
        sort_ta = torch.sort(model.odeblock.odefunc.t_a)[0]
        sort_ra = torch.sort(model.odeblock.odefunc.r_a)[0]
        spectrum_ax[row, 1].plot(range(L.shape[0]), sort_ta.cpu().numpy(), c='tab:green', label='t_a', zorder=10)
        spectrum_ax[row, 1].plot(range(L.shape[0]), sort_ra.cpu().numpy(), c='tab:orange', label='r_a', zorder=5)
        spectrum_ax[row, 1].legend()

    spectrum_ax[row, 1].set_title(f"Omega, E-values, E-vectors, epoch {epoch}", fontdict={'fontsize': 24})
    spectrum_ax[row, 1].xaxis.set_tick_params(labelsize=16)
    spectrum_ax[row, 1].yaxis.set_tick_params(labelsize=16)

    mat2 = spectrum_ax[row, 2].matshow(Q.cpu().numpy(), interpolation='nearest')
    spectrum_ax[row, 2].xaxis.set_tick_params(labelsize=16)
    spectrum_ax[row, 2].yaxis.set_tick_params(labelsize=16)
    cbar1 = spectrum_fig.colorbar(mat2, ax=spectrum_ax[row, 2], shrink=0.75)
    cbar1.ax.tick_params(labelsize=16)
    if not torch.cuda.is_available():
        spectrum_fig.show()

    ###2) multi grid accuracy and entropy charts
    train_accs = model.odeblock.odefunc.train_accs
    val_accs = model.odeblock.odefunc.val_accs
    test_accs = model.odeblock.odefunc.test_accs
    homophils = model.odeblock.odefunc.homophils

    acc_entropy_ax[row, 0].plot(np.arange(0.0, len(train_accs) * opt['step_size'], opt['step_size']), train_accs,
                                label="train")
    acc_entropy_ax[row, 0].plot(np.arange(0.0, len(val_accs) * opt['step_size'], opt['step_size']), val_accs,
                                label="val")
    acc_entropy_ax[row, 0].plot(np.arange(0.0, len(test_accs) * opt['step_size'], opt['step_size']), test_accs,
                                label="test")
    acc_entropy_ax[row, 0].plot(np.arange(0.0, len(homophils) * opt['step_size'], opt['step_size']), homophils,
                                label="homophil")
    acc_entropy_ax[row, 0].xaxis.set_tick_params(labelsize=16)
    acc_entropy_ax[row, 0].yaxis.set_tick_params(labelsize=16)
    acc_entropy_ax[row, 0].set_title(f"Accuracy evolution, epoch {epoch}", fontdict={'fontsize': 24})
    acc_entropy_ax[row, 0].legend(loc="upper right", fontsize=24)
    # acc_entropy_fig.show()

    # entropy plots #getting the line colour to change
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
    # https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection
    entropies = model.odeblock.odefunc.entropies

    # fig, ax = plt.subplots() #figsize=(8, 16))
    x = np.arange(0.0, entropies['entropy_train_mask'].shape[0] * opt['step_size'], opt['step_size'])
    ys = entropies['entropy_train_mask'].cpu().numpy()
    # ax.set_xlim(np.min(x), np.max(x))
    # ax.set_ylim(np.min(ys), np.max(ys))
    acc_entropy_ax[row, 1].set_xlim(np.min(x), np.max(x))
    acc_entropy_ax[row, 1].set_ylim(np.min(ys), np.max(ys))
    acc_entropy_ax[row, 1].xaxis.set_tick_params(labelsize=16)
    acc_entropy_ax[row, 1].yaxis.set_tick_params(labelsize=16)

    cmap = ListedColormap(['r', 'g'])
    norm = BoundaryNorm([-1, 0.5, 2.0], cmap.N)
    for i in range(entropies['entropy_train_mask'].shape[1]):
        tf = entropies['entropy_train_mask_correct'][:, i].float().cpu().numpy()
        points = np.expand_dims(np.concatenate([x.reshape(-1, 1),
                                                entropies['entropy_train_mask'][:, i].reshape(-1, 1).cpu().numpy()], axis=1), axis=1)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm)
        lc.set_array(tf[:-1])
        # ax.add_collection(lc)
        acc_entropy_ax[row, 1].add_collection(lc)

    # ax.set_title(f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}")
    acc_entropy_ax[row, 1].set_title(
        f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}",
        fontdict={'fontsize': 24})
    if not torch.cuda.is_available():
        acc_entropy_fig.show()

    ###3) multi grid edge value plots
    fOmf = model.odeblock.odefunc.fOmf
    edge_homophils = model.odeblock.odefunc.edge_homophils
    edge_evol_ax[row, 0].plot(np.arange(0.0, fOmf.shape[0] * opt['step_size'], opt['step_size']), fOmf.cpu().numpy())
    # too slow potentially can use pandas for the color map
    # colormap = cm.Spectral  # jet
    # normalize = mcolors.Normalize(vmin=edge_homophils.min(), vmax=edge_homophils.max())
    # for e in range(fOmf.shape[1]):
    #   color = colormap(normalize(edge_homophils[e]))
    #   edge_evol_ax[row,0].plot(np.arange(0.0, fOmf.shape[0] * opt['step_size'], opt['step_size']),
    #                          fOmf, c=color)
    edge_evol_ax[row, 0].xaxis.set_tick_params(labelsize=16)
    edge_evol_ax[row, 0].yaxis.set_tick_params(labelsize=16)
    edge_evol_ax[row, 0].set_title(f"fOmf, epoch {epoch}", fontdict={'fontsize': 24})

    L2dist = model.odeblock.odefunc.L2dist
    edge_evol_ax[row, 1].plot(np.arange(0.0, L2dist.shape[0] * opt['step_size'], opt['step_size']), L2dist.cpu().numpy())
    edge_evol_ax[row, 1].xaxis.set_tick_params(labelsize=16)
    edge_evol_ax[row, 1].yaxis.set_tick_params(labelsize=16)
    edge_evol_ax[row, 1].set_title(f"L2dist, epoch {epoch}", fontdict={'fontsize': 24})
    if not torch.cuda.is_available():
        edge_evol_fig.show()

    ###4) multi grid node evol plots
    #node magnitudes
    magnitudes = model.odeblock.odefunc.node_magnitudes
    node_evol_ax[row, 0].plot(np.arange(0.0, magnitudes.shape[0] * opt['step_size'], opt['step_size']), magnitudes.cpu().numpy())
    # labels  = model.odeblock.odefunc.labels
    # colormap = cm.Spectral  # jet
    # normalize = mcolors.Normalize(vmin=labels.min(), vmax=labels.max())
    # for n in range(magnitudes.shape[1]):
    #   color = colormap(normalize(labels[n]))
    #   node_evol_ax[row, 0].plot(np.arange(0.0, magnitudes.shape[0] * opt['step_size'], opt['step_size']),
    #                             magnitudes[:,n], color=color)
    node_evol_ax[row, 0].xaxis.set_tick_params(labelsize=16)
    node_evol_ax[row, 0].yaxis.set_tick_params(labelsize=16)
    node_evol_ax[row, 0].set_title(f"f magnitudes, epoch {epoch}", fontdict={'fontsize': 24})

    measures = model.odeblock.odefunc.node_measures
    node_evol_ax[row, 1].plot(np.arange(0.0, measures.shape[0] * opt['step_size'], opt['step_size']), measures.cpu().numpy())
    node_homophils = model.odeblock.odefunc.node_homophils
    # for n in range(measures.shape[1]):
    #   normalize = mcolors.Normalize(vmin=node_homophils.min(), vmax=node_homophils.max())
    #   colormap = cm.Spectral #jet
    #   color = colormap(normalize(node_homophils[n]))
    #   node_evol_ax[row, 1].plot(np.arange(0.0, measures.shape[0] * opt['step_size'], opt['step_size']),
    #                             measures[:,n], color=color)
    node_evol_ax[row, 1].xaxis.set_tick_params(labelsize=16)
    node_evol_ax[row, 1].yaxis.set_tick_params(labelsize=16)
    node_evol_ax[row, 1].set_title(f"Node measures, epoch {epoch}", fontdict={'fontsize': 24})
    # node_evol_fig.show()

    confusions = model.odeblock.odefunc.confusions
    colormap = cm.get_cmap(name="Set1")
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    conf_set = ['all','train', 'val', 'test']
    for i, conf_mat in enumerate(confusions):
        for c in range(conf_mat.shape[0]):
            correct = conf_mat[c,c,:]
            node_evol_ax[row, 2].plot(np.arange(0.0, correct.shape[0] * opt['step_size'], opt['step_size']),
                                  correct.cpu().numpy(), color=colormap(c), linestyle=linestyles[i], label=f"{conf_set[i]}_c{c}")
            # incorrect = torch.sum(conf_mat[c], dim=0) - correct #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html - sum over predictions j=cols
            # node_evol_ax[row, 1].plot(np.arange(0.0, incorrect.shape[0] * opt['step_size'], opt['step_size']),
            #                       incorrect, color=colormap(c), linestyle=linestyles[i], label=f"{conf_set[i]}_c{c}")

    node_evol_ax[row, 2].xaxis.set_tick_params(labelsize=16)
    node_evol_ax[row, 2].yaxis.set_tick_params(labelsize=16)
    node_evol_ax[row, 2].set_title(f"Correct preds evol, epoch {epoch}", fontdict={'fontsize': 24})
    node_evol_ax[row, 2].legend()
    if not torch.cuda.is_available():
        node_evol_fig.show()

    ###5) multi grid node scatter plots
    # node magnitude against degree or homophilly, colour is class
    magnitudes = model.odeblock.odefunc.node_magnitudes
    node_homophils = model.odeblock.odefunc.node_homophils
    labels = model.odeblock.odefunc.labels
    node_scatter_ax[row, 0].scatter(x=magnitudes[-1, :].cpu().numpy(), y=node_homophils.cpu().numpy(), c=labels.cpu().numpy())  # , cmap='Set1')
    node_scatter_ax[row, 0].xaxis.set_tick_params(labelsize=16)
    node_scatter_ax[row, 0].yaxis.set_tick_params(labelsize=16)
    node_scatter_ax[row, 0].set_title(f"f magnitudes v node homophils, epoch {epoch}", fontdict={'fontsize': 24})

    # node measure against degree or homophilly, colour is class
    measures = model.odeblock.odefunc.node_measures
    node_scatter_ax[row, 1].scatter(x=measures[-1, :].cpu().numpy(), y=node_homophils.cpu().numpy(), c=labels.cpu().numpy())  # , cmap='Set1')
    node_scatter_ax[row, 1].xaxis.set_tick_params(labelsize=16)
    node_scatter_ax[row, 1].yaxis.set_tick_params(labelsize=16)
    node_scatter_ax[row, 1].set_title(f"Node measures v node homophils, epoch {epoch}", fontdict={'fontsize': 24})
    if not torch.cuda.is_available():
        node_scatter_fig.show()

    ###6) scatter plot for edges
    # edge dot product against edge distance, coloured by edge homopholliy
    fOmf = model.odeblock.odefunc.fOmf
    L2dist = model.odeblock.odefunc.L2dist
    edge_homophils = model.odeblock.odefunc.edge_homophils
    mask = edge_homophils == 1
    edge_scatter_ax[row].scatter(x=fOmf[-1, :][mask].cpu().numpy(), y=L2dist[-1, :][mask].cpu().numpy(), label='edge:1',
                                 c='gold')  # c=edge_homophils[mask])
    edge_scatter_ax[row].scatter(x=fOmf[-1, :][~mask].cpu().numpy(), y=L2dist[-1, :][~mask].cpu().numpy(), label='edge:0',
                                 c='indigo')  # c=edge_homophils[~mask])

    edge_scatter_ax[row].xaxis.set_tick_params(labelsize=16)
    edge_scatter_ax[row].yaxis.set_tick_params(labelsize=16)
    edge_scatter_ax[row].set_title(f"Edge fOmf against L2, epoch {epoch}", fontdict={'fontsize': 24})
    edge_scatter_ax[row].legend(loc="upper right", fontsize=24)
    if not torch.cuda.is_available():
        edge_scatter_fig.show()

    ###7) class distances
    #column 0
    val_dist_mean_feat = model.odeblock.odefunc.val_dist_mean_feat
    val_dist_sd_feat = model.odeblock.odefunc.val_dist_sd_feat
    test_dist_mean_feat = model.odeblock.odefunc.test_dist_mean_feat
    test_dist_sd_feat = model.odeblock.odefunc.test_dist_sd_feat
    colormap = cm.get_cmap(name="Set1")
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for c in range(val_dist_mean_feat.shape[0]):
        # plot diags
        class_dist_ax[row, 0].plot(np.arange(0.0, val_dist_mean_feat.shape[-1] * opt['step_size'], opt['step_size']),
                                  val_dist_mean_feat[c,c,:].cpu().numpy(), color=colormap(c), linestyle=linestyles[0],
                                  label=f"base_{c}_eval_{c}")
        # output: rows base_class, cols eval_class
        for i in range(val_dist_mean_feat.shape[0]):
            class_dist_ax[row, 0].plot(np.arange(0.0, val_dist_mean_feat.shape[-1] * opt['step_size'], opt['step_size']),
                                       val_dist_mean_feat[i, c, :].cpu().numpy(), color=colormap(c),
                                       linestyle=linestyles[1],
                                       label=f"base_non{c}_eval_{c}")
    class_dist_ax[row, 0].xaxis.set_tick_params(labelsize=16)
    class_dist_ax[row, 0].yaxis.set_tick_params(labelsize=16)
    class_dist_ax[row, 0].set_title(f"Class feature L2 distances evol, epoch {epoch}", fontdict={'fontsize': 24})
    class_dist_ax[row, 0].legend()

    #column 1
    val_dist_mean_label = model.odeblock.odefunc.val_dist_mean_label
    for c in range(val_dist_mean_label.shape[0]):
        # plot diags
        class_dist_ax[row, 1].plot(np.arange(0.0, val_dist_mean_label.shape[-1] * opt['step_size'], opt['step_size']),
                                  val_dist_mean_label[c,c,:].cpu().numpy(), color=colormap(c), linestyle=linestyles[0],
                                  label=f"base_{c}_eval_{c}")
        # output: rows base_class, cols eval_class
        for i in range(val_dist_mean_label.shape[0]):
            class_dist_ax[row, 1].plot(np.arange(0.0, val_dist_mean_label.shape[-1] * opt['step_size'], opt['step_size']),
                                       val_dist_mean_label[i, c, :].cpu().numpy(), color=colormap(c),
                                       linestyle=linestyles[1],
                                       label=f"base_non{c}_eval_{c}")
    class_dist_ax[row, 1].xaxis.set_tick_params(labelsize=16)
    class_dist_ax[row, 1].yaxis.set_tick_params(labelsize=16)
    class_dist_ax[row, 1].set_title(f"Class label L2 distances evol, epoch {epoch}", fontdict={'fontsize': 24})
    class_dist_ax[row, 1].legend()

    if not torch.cuda.is_available():
        class_dist_fig.show()

    model.odeblock.odefunc.fOmf = None
    model.odeblock.odefunc.attentions = None
    model.odeblock.odefunc.L2dist = None
    model.odeblock.odefunc.node_magnitudes = None
    model.odeblock.odefunc.node_measures = None
    model.odeblock.odefunc.train_accs = None
    model.odeblock.odefunc.val_accs = None
    model.odeblock.odefunc.test_accs = None
    model.odeblock.odefunc.entropies = None

    model.odeblock.odefunc.val_dist_mean_feat = None
    model.odeblock.odefunc.val_dist_sd_feat = None
    model.odeblock.odefunc.test_dist_mean_feat = None
    model.odeblock.odefunc.test_dist_sd_feat = None
    model.odeblock.odefunc.val_dist_mean_label = None
    model.odeblock.odefunc.val_dist_sd_label = None
    model.odeblock.odefunc.test_dist_mean_label = None
    model.odeblock.odefunc.test_dist_sd_label = None

    #todo if flagged to save reports in wandb, then save every full page as a jpeg in wandb
    if (row == num_rows - 1 or epoch == opt['wandb_epoch_list'][-1]) and opt['save_local_reports']:
        model.odeblock.odefunc.spectrum_pdf.savefig(spectrum_fig)
        model.odeblock.odefunc.acc_entropy_pdf.savefig(acc_entropy_fig)
        model.odeblock.odefunc.edge_evol_pdf.savefig(edge_evol_fig)
        model.odeblock.odefunc.node_evol_pdf.savefig(node_evol_fig)
        model.odeblock.odefunc.node_scatter_pdf.savefig(node_scatter_fig)
        model.odeblock.odefunc.edge_scatter_pdf.savefig(edge_scatter_fig)
        model.odeblock.odefunc.class_dist_pdf.savefig(class_dist_fig)

    if (row == num_rows - 1 or epoch == opt['wandb_epoch_list'][-1]) and opt['save_wandb_reports']:
        wandb.log({f"spectrum_fig_{idx//num_rows}": wandb.Image(spectrum_fig),
            f"acc_entropy_fig_{idx//num_rows}": wandb.Image(acc_entropy_fig),
            f"edge_evol_fig_{idx//num_rows}": wandb.Image(edge_evol_fig),
            f"node_evol_fig_{idx//num_rows}": wandb.Image(node_evol_fig),
            f"node_scatter_fig_{idx//num_rows}": wandb.Image(node_scatter_fig),
            f"edge_scatter_fig_{idx//num_rows}": wandb.Image(edge_scatter_fig),
            f"class_dist_fig_{idx // num_rows}": wandb.Image(class_dist_fig)})

    if epoch == opt['wandb_epoch_list'][-1] and opt['save_local_reports']:
        model.odeblock.odefunc.spectrum_pdf.close()
        model.odeblock.odefunc.acc_entropy_pdf.close()
        model.odeblock.odefunc.edge_evol_pdf.close()
        model.odeblock.odefunc.node_evol_pdf.close()
        model.odeblock.odefunc.node_scatter_pdf.close()
        model.odeblock.odefunc.edge_scatter_pdf.close()
        model.odeblock.odefunc.class_dist_pdf.close()


def report_1(ax, fig, odefunc, row, epoch):
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

    ax[row, 1].set_title(f"Omega, E-values, E-vectors, epoch {epoch}, block {odefunc.block_num}", fontdict={'fontsize': 24})
    ax[row, 1].xaxis.set_tick_params(labelsize=16)
    ax[row, 1].yaxis.set_tick_params(labelsize=16)

    mat2 = ax[row, 2].matshow(Q.cpu().numpy(), interpolation='nearest')
    ax[row, 2].xaxis.set_tick_params(labelsize=16)
    ax[row, 2].yaxis.set_tick_params(labelsize=16)
    cbar1 = fig.colorbar(mat2, ax=ax[row, 2], shrink=0.75)
    cbar1.ax.tick_params(labelsize=16)
    if not torch.cuda.is_available():
        fig.show()


def report_2(ax, fig, odefunc, row, epoch):
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
        lc.set_array(tf[:-1])
        # ax.add_collection(lc)
        ax[row, 1].add_collection(lc)

    # ax.set_title(f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}")
    ax[row, 1].set_title(
        f"Train set (num_nodes {entropies['entropy_train_mask'].shape[1]}) Entropy, epoch {epoch}, block {odefunc.block_num}",
        fontdict={'fontsize': 24})
    if not torch.cuda.is_available():
        fig.show()

def report_3(ax, fig, odefunc, row, epoch):
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
    if not torch.cuda.is_available():
        fig.show()

def report_4(ax, fig, odefunc, row, epoch):
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
    if not torch.cuda.is_available():
        fig.show()

def report_5(ax, fig, odefunc, row, epoch):
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
    if not torch.cuda.is_available():
        fig.show()

def report_6(ax, fig, odefunc, row, epoch):
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
    if not torch.cuda.is_available():
        fig.show()

def report_7(ax, fig, odefunc, row, epoch):
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

    if not torch.cuda.is_available():
        fig.show()

def run_reports_lie_trotter(odefunc):
    opt = odefunc.opt
    epoch = odefunc.epoch

    # find position of current epoch in epoch list
    idx = opt['wandb_epoch_list'].index(epoch)
    # determine index % num per page
    num_rows = 4
    row = idx % num_rows

    # self.pdf_list = ['spectrum', 'acc_entropy', 'edge_evol', 'node_evol', 'node_scatter', 'edge_scatter']
    # fig_dims = {1: ['spectrum', 3, [1, 1, 1], (24, 32)],
    #             2: ['acc_entropy', 2, [1, 1], (24, 32)],
    #             3: ['edge_evol', 2, [1, 1], (24, 32)],
    #             4: ['node_evol', 3, [1, 1, 2], (24, 32)],
    #             5: ['node_scatter', 2, [1, 1], (24, 32)],
    #             6: ['edge_scatter', 1, [1], (24, 32)],
    #             7: ['class_dist', 2, [1, 1], (24, 32)]}

    reports_nums = opt['reports_list']
    for rep_num in reports_nums:
        report_num = globals()[f"report_{rep_num}"]

        if row == 0:  # create new figs and pdfs
            desc, cols, widths, fig_size = odefunc.opt['fig_dims'][rep_num]
            fig, ax = plt.subplots(num_rows, cols, gridspec_kw={'width_ratios': widths}, figsize=fig_size)
            if hasattr(odefunc, f"report{str(rep_num)}_fig_list"):
                odefunc.getattr(f"report{str(rep_num)}_fig_list").append([fig, ax])
            else:
                setattr(odefunc, f"report{str(rep_num)}_fig_list", [[fig, ax]])
                savefolder = f"./plots/{opt['gnl_savefolder']}"
                setattr(odefunc, f"report{str(rep_num)}_pdf", PdfPages(f"{savefolder}/{desc}.pdf"))
        else:
            fig, ax = getattr(odefunc, f"report{str(rep_num)}_fig_list")[-1]

        report_num(ax, fig, odefunc, row, epoch)

    #save fig to pdf
    if (row == num_rows - 1 or epoch == opt['wandb_epoch_list'][-1]) and opt['save_local_reports']:
        odefunc.getattr(f"report{str(rep_num)}_pdf").savefig(fig)
    #log fig to wandb
    if (row == num_rows - 1 or epoch == opt['wandb_epoch_list'][-1]) and opt['save_wandb_reports']:
        wandb.log({f"report{str(rep_num)}_fig_{idx // num_rows}": wandb.Image(fig)})
    #close pdf
    if epoch == opt['wandb_epoch_list'][-1] and opt['save_local_reports']:
        odefunc.getattr(f"report{str(rep_num)}_pdf").close()

    reset_stats(odefunc)

    # ###1) multi grid Omega spectrum charts
    # report_1(spectrum_ax, spectrum_fig, odefunc, opt, row, epoch)
    # ###2) multi grid accuracy and entropy charts
    # report_2(spectrum_ax, spectrum_fig, odefunc, opt, row, epoch)
    # ###3) multi grid edge value plots
    # report_3(edge_evol_ax, edge_evol_fig, odefunc, opt, row, epoch)
    # ###4) multi grid node evol plots
    # report_4(node_evol_ax, node_evol_fig, odefunc, opt, row, epoch)
    # ###5) multi grid node scatter plots
    # report_5(node_scatter_ax,  node_scatter_fig, odefunc, opt, row, epoch)
    # ###6) scatter plot for edges
    # report_6(edge_scatter_ax, edge_scatter_fig, odefunc, opt, row, epoch)
    # ###7) class distances
    # report_7(class_dist_ax, edge_scatter_fig, odefunc, opt, row, epoch)


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

def reports_manager(model, data):
    # forward pass through the model in eval mode to generate the data
    model.odeblock.odefunc.get_evol_stats = True
    pred = model.forward(data.x).max(1)[1]
    model.odeblock.odefunc.get_evol_stats = False

    for func in model.odeblock.funcs:
        func.epoch = model.odeblock.odefunc.epoch
        func.get_evol_stats = model.odeblock.odefunc.get_evol_stats
        func.wandb_step = model.odeblock.odefunc.wandb_step
        run_reports_lie_trotter(func)

