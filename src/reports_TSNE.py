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

def project_label_space(m2, X):
    '''converts 3 tensor iin feature spaace into label space'''
    X = torch.from_numpy(X)#nodes x features x time
    X = X.permute(dims=[0, 2, 1])
    Y = X.reshape(-1, X.shape[-1])     #reshape to be (nodes*time) x features
    M = m2(Y)
    L = M.reshape(X.shape[0], -1, M.shape[-1])     #reverse reshape to be nodes x features x time
    L = L.permute(dims=[0, 2, 1])
    return L.detach().numpy()

#old style report 8 TSNE where not using wandb logging and artifacts
# def tsne_snap_old(ax, fig, odefunc, row, epoch, savefolder, s=None):
#     '''function called by report_8 to maintain syntax of "online" reporting
#     for a particular epoch will generate fig for the PDF Sheets workflow, ie 4 columns'''
#
#     npy_path = savefolder + f"/paths_epoch{epoch}_{odefunc.opt['dataset']}.npy"
#     npy_label = savefolder + f"/labels_epoch{epoch}_{odefunc.opt['dataset']}.npy"
#     m2_path =  savefolder + f"/m2_epoch{epoch}_{odefunc.opt['dataset']}.pt"
#
#     X = np.load(npy_path)
#     m2 = torch.load(m2_path)
#     X = project_label_space(m2, X)
#     labels = np.load(npy_label) #=odefunc.labels.cpu().numpy()
#     # load paths np array
#     TL = X.shape[-1]
#     X0pca2 = X0_PCA(X) #2D basis for TSNE inits
#
#     nr, nc = 4, 4
#     dt_idx = TL // (nc - 1)
#     idx_list = [0] + list(range(dt_idx, dt_idx*(nc-1), dt_idx)) + [TL-1]
#     times = odefunc.cum_time_ticks
#     block_types = odefunc.block_type_list
#
#     for i, t_idx in enumerate(idx_list):
#         X_emb = Xt_TSNE(X, X0pca2, t_idx)
#         if s:
#             ax[row, i].scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=labels, s=s)
#         else:
#             ax[row, i].scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=labels)
#         ax[row, i].xaxis.set_tick_params(labelsize=16)
#         ax[row, i].yaxis.set_tick_params(labelsize=16)
#         if odefunc.opt['lie_trotter'] == 'gen_2':
#             ax[row, i].set_title(f"{odefunc.opt['dataset']} TSNE, e={epoch}, t={odefunc.cum_time_ticks[t_idx]}, {block_types[t_idx]} ", fontdict={'fontsize': 24})
#         else:
#             ax[row, i].set_title(f"{odefunc.opt['dataset']} TSNE, e={epoch}, t={times[t_idx]}", fontdict={'fontsize': 24})
#
#         ax[row, i].legend(loc="upper right", fontsize=24)
#     if not torch.cuda.is_available():
#         fig.show()

def tsne_snap(ax, fig, odefunc, row, epoch, s=None):
    '''function called by report_8 to maintain syntax of "online" reporting
    for a particular epoch will generate fig for the PDF Sheets workflow, ie 4 columns'''

    X = np.stack(odefunc.paths, axis=-1)
    labels = np.stack(odefunc.labels, axis=-1)
    G = to_networkx(odefunc.data) #for optional including of the graph
    m2 = odefunc.GNN_m2
    X = project_label_space(m2, X)
    #todo account for centers in either feature or label space
    centers = np.eye(odefunc.C)[...,np.newaxis].repeat(repeats=X.shape[-1], axis=-1)

    TL = X.shape[-1]
    X0pca2 = X0_PCA(X) #2D basis for TSNE inits

    nr, nc = 4, 4
    dt_idx = TL // (nc - 1)
    idx_list = [0] + list(range(dt_idx, dt_idx*(nc-1), dt_idx)) + [TL-1]
    times = odefunc.cum_time_ticks
    block_types = odefunc.block_type_list

    for i, t_idx in enumerate(idx_list):
        X_emb = Xt_TSNE(X, X0pca2, t_idx)
        if s:
            ax[row, i].scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=labels, s=s)
        else:
            ax[row, i].scatter(x=X_emb[:, 0], y=X_emb[:, 1], c=labels)

        c_i = Xt_TSNE(centers, X0pca2, i)
        ax[row, i].scatter(c_i[:,0], c_i[:,1], c='r', s=80, marker='x', linewidths=5, zorder=10)

        ax[row, i].xaxis.set_tick_params(labelsize=16)
        ax[row, i].yaxis.set_tick_params(labelsize=16)
        if odefunc.opt['lie_trotter'] == 'gen_2':
            ax[row, i].set_title(f"{odefunc.opt['dataset']} TSNE, e={epoch}, t={odefunc.cum_time_ticks[t_idx]}, {block_types[t_idx]} ", fontdict={'fontsize': 24})
        else:
            ax[row, i].set_title(f"{odefunc.opt['dataset']} TSNE, e={epoch}, t={times[t_idx]}", fontdict={'fontsize': 24})

        # ax[row, i].legend(loc="upper right", fontsize=24)
    if not torch.cuda.is_available() and epoch in odefunc.opt['display_epoch_list']:
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


def tsne_ani(gnl_savefolder, dataset, epoch, s=None):
    '''function to generate TSNE plots in animation for every timestep '''
    savefolder = f"../plots/{gnl_savefolder}_{dataset}"
    npy_path = savefolder + f"/paths_epoch{epoch}_{dataset}.npy"
    npy_label = savefolder + f"/labels_epoch{epoch}_{dataset}.npy"
    m2_path =  savefolder + f"/m2_epoch{epoch}_{dataset}.pt"
    X = np.load(npy_path)
    m2 = torch.load(m2_path)
    X = project_label_space(m2, X)

    X0pca2 = X0_PCA(X)
    labels = np.load(npy_label)
    NXgraph = nx.read_gpickle(savefolder + f"/nxgraph_epoch{epoch}_{dataset}.pkl")
    # opt = json.load(savefolder + 'opt.json')
    params = {'fps': 1, 'node_size': 100, 'edge_width': 0.25, 'im_height': 8, 'im_width': 16}
    create_animation(X, X0pca2, labels, NXgraph, params, savefolder, dataset, epoch)


def create_animation(X, X0pca2, labels, NXgraph, params, savefolder, dataset, epoch):
    # loop through data and update plot
    def update(ii, X, X0pca2, labels, ax, NXgraph, params):
        plt.tight_layout()
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        # ax.set_xlim([-3, 3])
        # ax.set_ylim([-3, 3])

        C = labels.max()
        centers = np.eye(X0pca2.shape[-1])[...,np.newaxis].repeat(repeats=X.shape[-1], axis=-1)
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
