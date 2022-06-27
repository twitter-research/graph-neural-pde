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
    if not torch.cuda.is_available():
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
    if not torch.cuda.is_available():
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
    if not torch.cuda.is_available():
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
    if not torch.cuda.is_available():
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
    if not torch.cuda.is_available():
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
    if not torch.cuda.is_available():
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

    if not torch.cuda.is_available():
        fig.show()

# def tsne_evol(ax, fig, odefunc, row, epoch):
def report_8(ax, fig, odefunc, row, epoch):
    # tsne_evol
    savefolder = f"../plots/{odefunc.opt['gnl_savefolder']}_{odefunc.opt['dataset']}"
    npy_path = savefolder + f"/paths_epoch{epoch}_{odefunc.opt['dataset']}.npy"
    npy_label = savefolder + f"/labels_epoch{epoch}_{odefunc.opt['dataset']}.npy"
    np.save(npy_path, np.stack(odefunc.paths, axis=-1))
    np.save(npy_label, np.stack(odefunc.labels, axis=-1))
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

    tsne_snap(ax, fig, odefunc, row, epoch, npy_path, npy_label)

'''Note on approach to TSNE / PCA visualisation
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
"It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) 
to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high"

So rather than just running PCA on the feature propagation tensor NxdxT, we project onto the top 2 PCA of the X0
This gives an embedding in n_compents=2 which is mirrors the process of selecting PCA initialisation

'''

def X0_PCA(X):
    '''
    use PCA initialisation to embed into 2d, however use PCA basis from X0 (t=0)
    https://github.com/scikit-learn/scikit-learn/blob/80598905e517759b4696c74ecc35c6e2eb508cff/sklearn/manifold/_t_sne.py#L981
    '''
    n_components = 2
    seed = 123
    random_state = check_random_state(seed)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    X0_embedded = pca.fit_transform(X[:, :, 0]).astype(np.float32, copy=False)
    X0pca = pca.components_#[:n_components, :]
    return X0pca

def Xt_TSNE(X, X0pca, t_idx):
    X_slice = X[:, :, t_idx]
    Xi_pca0emb = X_slice @ X0pca.T
    # X_emb = TSNE(n_components=2, learning_rate='auto', init=Xi_pca0emb).fit_transform(Xi_pca0emb) - this is trivial
    #use X_slice as the high dimensional data and Xi_pca0emb as the low dim initialisation
    # - this means high dimensional distances/realtionships are reflected
    X_emb = TSNE(n_components=2, learning_rate='auto', init=Xi_pca0emb).fit_transform(X_slice)
    #todo https://discuss.pytorch.org/t/t-sne-for-pytorch/44264
    return X_emb

def project_label_space(path, X):
    '''converts 3 tensor iin feature spaace into label space'''
    m2 = torch.load(path)
    X = torch.from_numpy(X)#nodes x features x time
    X = X.permute(dims=[0, 2, 1])
    Y = X.reshape(-1, X.shape[-1])     #reshape to be (nodes*time) x features
    M = m2(Y)
    L = M.reshape(X.shape[0], -1, M.shape[-1])     #reverse reshape to be nodes x features x time
    L = L.permute(dims=[0, 2, 1])
    return L.detach().numpy()


def tsne_snap(ax, fig, odefunc, row, epoch, npy_path, npy_label, s=None):
    '''function called by report_8 to maintain syntax of "online" reporting
    for a particular epoch will generate fig for the PDF Sheets workflow, ie 4 columns'''
    X = np.load(npy_path)
    labels = np.load(npy_label) #=odefunc.labels.cpu().numpy()
    # load paths np array
    TL = X.shape[-1]
    X0pca2 = X0_PCA(X) #2D basis for TSNE inits

    nr, nc = 4, 4
    dt_idx = TL // nc
    idx_list = [0] + list(range(dt_idx, dt_idx*(nc-1), dt_idx)) + [TL-1]

    for i, t_idx in enumerate(idx_list):
        X_emb = Xt_TSNE(X, X0pca2, t_idx)
        if s:
            ax[row, i].scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=labels, s=s)
        else:
            ax[row, i].scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=labels)
        ax[row, i].xaxis.set_tick_params(labelsize=16)
        ax[row, i].yaxis.set_tick_params(labelsize=16)
        if odefunc.opt['lie_trotter'] == 'gen_2':
            ax[row, i].set_title(f"{odefunc.opt['dataset']} TSNE, e={epoch}, t={times[t_idx]}, {block_types[t_idx]} ", fontdict={'fontsize': 24})
        else:
            ax[row, i].set_title(f"{odefunc.opt['dataset']} TSNE, e={epoch}, t={times[t_idx]}", fontdict={'fontsize': 24})

        ax[row, i].legend(loc="upper right", fontsize=24)
    if not torch.cuda.is_available():
        fig.show()

def tsne_full(gnl_savefolder, dataset, epoch, cols, s=None):
    '''function to generate TSNE plots in PDF pages for every timestep
    currently does not save the PdfPages'''
    savefolder = f"../plots/{gnl_savefolder}_{dataset}"
    npy_path = savefolder + f"/paths_epoch{epoch}_{dataset}.npy"
    npy_label = savefolder + f"/labels_epoch{epoch}_{dataset}.npy"
    X = np.load(npy_path)
    labels = np.load(npy_label)

    T = X.shape[-1]
    max_rows = 4
    num_rows = T // cols + 1
    widths = cols * [1]
    fig_size = (24, 32)

    #create pdf
    PdfPages(savefolder + f"tsne_full_epoch{epoch}_{dataset}")
    #do t=0 PCA
    X0pca2 = X0_PCA(X)

    fig, ax = plt.subplots(max_rows, cols, gridspec_kw={'width_ratios': widths}, figsize=fig_size)
    for i in range(T):
        row = (i // cols) % max_rows
        col = i % cols

        X_emb = Xt_TSNE(X, X0pca2, i)
        if s:
            ax[row, col].scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=labels, s=s)
        else:
            ax[row, col].scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=labels)
        ax[row, col].xaxis.set_tick_params(labelsize=16)
        ax[row, col].yaxis.set_tick_params(labelsize=16)
        ax[row, col].set_title(f"{dataset} TSNE, epoch {epoch}, time {i/(T-1):.2f}", fontdict={'fontsize': 24})
        ax[row, col].legend(loc="upper right", fontsize=24)

        if row == max_rows - 1 and col == cols - 1:  # create new figs and pdfs
            if not torch.cuda.is_available():
                fig.show()
            # create fig
            fig, ax = plt.subplots(max_rows, cols, gridspec_kw={'width_ratios': widths}, figsize=fig_size)

    if not torch.cuda.is_available():
        fig.show()

def tsne_ani(gnl_savefolder, dataset, epoch, cols, s=None):
    '''function to generate TSNE plots in animation for every timestep '''
    savefolder = f"../plots/{gnl_savefolder}_{dataset}"
    npy_path = savefolder + f"/paths_epoch{epoch}_{dataset}.npy"
    npy_label = savefolder + f"/labels_epoch{epoch}_{dataset}.npy"
    m2_path =  savefolder + f"/m2_epoch{epoch}_{dataset}.pt"
    X = np.load(npy_path)
    X = project_label_space(m2_path, X)

    X0pca2 = X0_PCA(X)
    labels = np.load(npy_label)
    NXgraph = nx.read_gpickle(savefolder + f"/nxgraph_epoch{epoch}_{dataset}.pkl")
    # opt = json.load(savefolder + 'opt.json')
    params = {'fps': 2, 'node_size': 100, 'edge_width': 0.25, 'im_height': 8, 'im_width': 16}
    create_animation(X, X0pca2, labels, NXgraph, params, savefolder, dataset, epoch)


def create_animation(X, X0pca2, labels, NXgraph, params, savefolder, dataset, epoch):
    # loop through data and update plot
    def update(ii, X, X0pca2, labels, ax, NXgraph, params):
        plt.tight_layout()
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        # ax.set_xlim([-3, 3])
        # ax.set_ylim([-3, 3])

        #add L coordinates for centers in label space
        centers = np.eye(X0pca2.shape[-1])[...,np.newaxis].repeat(repeats=X.shape[-1], axis=-1)
        C = labels.max()
        # X = np.concatenate([X,centers], axis=0)
        # labels = np.concatenate([labels, np.array(X0pca2.shape[-1] * [C])])

        pos_0 = Xt_TSNE(X, X0pca2, 0)
        ax.clear()
        pos_i = Xt_TSNE(X, X0pca2, ii) #X_emb[:, :, ii]
        pos_i_dict = {i: pos_i[i, :].tolist() for i in range(pos_0.shape[0])}
        nx.draw(NXgraph, pos=pos_i_dict, ax=ax, node_size=params['node_size'],
                node_color=labels, cmap=plt.get_cmap('Spectral'), arrows=False, width=0.25)#, zorder=0)  # =params['edge_with'] )

        c_i = Xt_TSNE(centers, X0pca2, ii)
        #https://stackoverflow.com/questions/37246941/specifying-the-order-of-matplotlib-layers
        ax.scatter(c_i[:,0], c_i[:,1], c='r', s=80, marker='x', linewidths=5, zorder=10)

        limits = plt.axis('on')
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(f"GRAFF evolution, diffusion idx={ii}", fontsize=16)

    _, ax = plt.subplots(figsize=(params['im_width'], params['im_height']))
    fig = plt.gcf()
    n_frames = X.shape[-1]
    animation = FuncAnimation(fig, func=update, frames=n_frames, fargs=(X, X0pca2, labels, ax, NXgraph, params))
    animation.save(savefolder + f"/ani_epoch{epoch}_{dataset}.gif", fps=params['fps'])  #writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=fps)


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
                8: ['tsne_evol', 4, [1, 1, 1, 1], (32,32)]}

    reports_nums = opt['reports_list']
    for rep_num in reports_nums:
        report_func = globals()[f"report_{rep_num}"]

        if row == 0:  # create new figs and pdfs
            desc, cols, widths, fig_size = odefunc.opt['fig_dims'][rep_num]
            fig, ax = plt.subplots(num_rows, cols, gridspec_kw={'width_ratios': widths}, figsize=fig_size)
            if hasattr(odefunc, f"report{str(rep_num)}_fig_list"):
                getattr(odefunc, f"report{str(rep_num)}_fig_list").append([fig, ax])
            else:
                setattr(odefunc, f"report{str(rep_num)}_fig_list", [[fig, ax]])
                # savefolder = f"../plots/{opt['gnl_savefolder']}"
                savefolder = f"../plots/{opt['gnl_savefolder']}_{opt['dataset']}"
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


if __name__ == "__main__":
    gnl_savefolder = 'tsne_evol'
    dataset = 'Cora'
    epoch = 128
    cols = 2
    # tsne_full(gnl_savefolder, dataset, epoch, cols, s=120)
    # ani()
    tsne_ani(gnl_savefolder, dataset, epoch, cols, s=None)
