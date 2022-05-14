import sys
import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d, axes3d
from matplotlib.patches import FancyArrowPatch
from pylab import rcParams
import torch
from torch_geometric.utils import to_dense_adj, to_undirected
from utils import DummyData


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

    #got this from fix in graphCon fancyarrowpatch.py
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M) #Transform the points by the projection matrix https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.mplot3d.proj3d.proj_transform.html
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

# class axes3d_nonsquare(axes3d):
#     #todo https://github.com/matplotlib/matplotlib/issues/8593
#     pass

def get_data(edge_index, x):
    num_nodes = 4
    num_classes = 2
    y = torch.tensor([0, 1, 0, 1])
    data = DummyData(edge_index, None, num_nodes, x, y, num_classes)
    return data

def get_dynamics(data, W, T, dt):
    edge_index = to_undirected(data.edge_index)
    adj = to_dense_adj(edge_index).squeeze()
    deginvsq = torch.pow(torch.sqrt(adj.sum(dim=0)), -1)
    A = torch.diag(deginvsq) @ adj @ torch.diag(deginvsq)
    A_eval, A_evec = torch.linalg.eigh(A)
    AW = torch.kron(A, W)
    AW_eval, AW_evec = torch.linalg.eigh(AW)
    print(f"A {A}")
    print(f"A_eval {A_eval}")
    print(f"A_evec (per row) {A_evec.T}")
    print(f"AW {AW}")
    print(f"AW_evec {AW_evec}")
    print(f"AW_eval (per row) {AW_eval.T}")
    x = data.x
    X = x.unsqueeze(dim=-1)
    Y = torch.zeros(X.shape, dtype=torch.float)
    for t in range(math.ceil(T/dt)):
        x = x + dt * A @ x @ W
        X = torch.cat((X, x.unsqueeze(dim=-1)), dim=-1)
        Y = torch.cat((Y, (dt * A @ x @ W).unsqueeze(dim=-1)), dim=-1)
    return X.detach().numpy(), Y.detach().numpy()

def plot_eigs(eig_vecs, eig_vals, clist, ax, nodes=True, edges=True):
    t = -0.70
    for j in range(2):
        xpos = 0.
        x1pos = eig_vecs[0, j] * eig_vals[0]
        ypos = 0.
        y1pos = eig_vecs[1, j] * eig_vals[1]
        arw = Arrow3D([t, t], [xpos, x1pos], [ypos, y1pos], arrowstyle="->", color=clist[j], lw=2, mutation_scale=25)
        ax.add_artist(arw)
        ax.text(t, x1pos * 1.2, y1pos * 1.2, f"e{j+1}", c=clist[j])

def plot_labels(labels, offset, X_all, clist, t_idx, T, ax):
    N = X_all.shape[0]
    num_T = X_all.shape[-1]
    #plot initial labels
    for n in range(N):
        ax.text(T, X_all[n,0,t_idx]*offset[n], X_all[n,1,t_idx]*offset[n], f"{labels[n]}", c=clist[n], size=14)

def plot_slices(num_slices, dt, edge_index, X_all, Y_all, W, clist, ax, nodes=True, edges=True, arrows=True, trace=False):
    edge_index = edge_index.detach().numpy()
    N = X_all.shape[0]
    T = X_all.shape[-1]
    t_idxs = [0] + list(range(T//num_slices, T - T//num_slices, T//num_slices)) + [T-1]
    #list(range(T)) #possbile here to control slice frequency

    #plot slices
    for t in t_idxs:
        pos = {i: [t*dt, X_all[i, 0, t].item(), X_all[i, 1, t].item()]
               for i in range(N)}
        node_xyz = np.array([pos[i] for i in range(N)])
        edge_xyz = np.array([(pos[uv[0]], pos[uv[1]]) for uv in edge_index.T])
        plot_3d(node_xyz, edge_xyz, clist, ax, nodes, edges)

        if arrows:
            colors = ["red","green","orange","blue"]
            if t in list(range(T-1)):
                for n in range(N):
                    xpos = X_all[n, 0, t]
                    x1pos = X_all[n, 0, t+T//num_slices]
                    ypos = X_all[n, 1, t]
                    y1pos = X_all[n, 1, t+T//num_slices]
                    # print(n, xpos,ypos,colors[n])
                    arw = Arrow3D([t*dt,t*dt],[xpos,x1pos],[ypos,y1pos], arrowstyle="->", color=colors[n], lw = 2, mutation_scale=25)
                    ax.add_artist(arw)
            else:
                pass
                # for n in range(N):
                #     xpos = X_all[n, 0, -1]
                #     x1pos = xpos + Y_all[n, 0, -1] * T//num_slices * dt
                #     ypos = X_all[n, 1, -1]
                #     y1pos = ypos + Y_all[n, 1, -1] * T//num_slices * dt
                #     # print(n, xpos,ypos,colors[n])
                #     arw = Arrow3D([t*dt,t*dt],[xpos,x1pos],[ypos,y1pos], arrowstyle="->", color=colors[n], lw = 2, mutation_scale=25)
                #     ax.add_artist(arw)

    # #plot_node_paths
    if trace:
        for n in range(N):
            ax.scatter(dt*np.linspace(0, T, T), X_all[n, 0, :], X_all[n, 1, :],
                   s=10.0, color=colors[n], ec="w")#, cmap=cmap)

    # cmap = plt.get_cmap("tab10")
    # # cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # for i in range(len(G.nodes)):
    #     # ax.scatter(X_all[i, :, 0], np.linspace(0, nlayers, nlayers + 1), X_all[i, :, 1],
    #     #            s=10.0, c=cycle[i], ec="w")
    #     ax.scatter(X_all[i, :, 0], np.linspace(0, nlayers, nlayers + 1), X_all[i, :, 1],
    #                s=10.0, color=cmap(i), ec="w")#, cmap=cmap)
    #
    # #plot_edges
    # edge = 2
    # for t in range(0, nlayers, 1):
    #     u, v = list(G.edges())[edge]
    #     ax.plot((X_all[u, t, 0].item(), X_all[v, t, 0].item()),
    #             (t, t),
    #             (X_all[u, t, 1].item(), X_all[v, t, 1].item()),
    #             color="tab:gray")

# https://networkx.org/documentation/stable/auto_examples/3d_drawing/plot_basic.html
def plot_3d(node_xyz, edge_xyz, clist, ax, nodes, edges):
    if nodes:
        # Plot the nodes - alpha is scaled by "depth" automatically
        cmap = plt.get_cmap("tab10")
        ax.scatter(*node_xyz.T, c=clist, s=80)#cmap=cmap#, ec="w")
    if edges:
        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        # ax.grid(False)
        # Suppress tick labels
        # for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        #     dim.set_ticks([])

        # ax.set_axis_off()

        # ax.xaxis.set_ticks([])
        # ax.yaxis.set_ticks([])
        # ax.zaxis.set_ticks([])
        # Set axes labels
        # ax.set_xlabel('time', fontsize=10)
        # ax.set_ylabel('x0-axis', fontsize=10)
        # ax.set_zlabel('x1-axis', fontsize=10)
    _format_axes(ax)


def plot_greed(fig=None, ax=None, ax_idx=None, plot=False, save=False):
    if ax is None:
        fig = plt.figure()#figsize=(12,8))
        ax = fig.add_subplot(111, projection="3d")

    #plot frame
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                               [1, 3, 0, 2, 1, 3, 0, 2]], dtype=torch.long)
    x = 2.0 * torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=torch.float)
    invsqrt2 = 1 / torch.sqrt(torch.tensor(2))
    eig_vecs = invsqrt2 * torch.tensor([[1, 1],[1, -1]], dtype=torch.float)
    # eig_vecs = torch.tensor([[1, 0],[0, 1]], dtype=torch.float)

    eig_vals = 0.95 * torch.tensor([-1.0, 1.0], dtype=torch.float)
    W = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
    T = 3.0
    dt = 0.1
    num_slices = 2#3 #excluding initial condition
    data = get_data(edge_index, x)
    X_all, Y_all = get_dynamics(data, W, T, dt)
    clist= 4*["grey"]
    plot_slices(num_slices, dt, edge_index, 2*X_all, Y_all, W, clist, ax, nodes=False, edges=True, arrows=False)

    #plot graph
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],#, 0, 2],
                               [1, 3, 0, 2, 1, 3, 0, 2]],#, 2, 0]],
                                dtype=torch.long)
    x = torch.tensor([[-1, -1], [-1, 1], [1.5, 2], [1, -1]], dtype=torch.float)
    # x = torch.tensor([[-1, -0.25], [-0.25, -1], [1.25, 0], [0, 1.25]], dtype=torch.float)
    data = get_data(edge_index, x)
    X_all, Y_all = get_dynamics(data, W, T, dt)
    clist = ["red","green","orange","blue"]
    plot_slices(num_slices, dt, edge_index, X_all, Y_all, W, clist, ax, nodes=True, edges=True, arrows=True, trace=True)
    #plot_labels
    offset = [1.8,2.2,1.2,1.9]
    clist = 4 * ["black"]
    plot_labels(data.y, offset, X_all, clist, 0, 0, ax)
    # plot_labels(data.y, offset, X_all, clist, -1, T, ax)
    clist = ["tab:blue", "tab:purple"]
    #plot eigs
    # plot_eigs(eig_vecs, eig_vals, clist, ax, nodes=True, edges=True)

    # ax.set_xlabel('time', fontsize=10)
    # ax.set_ylabel('x0-axis', fontsize=10)
    # ax.set_zlabel('x1-axis', fontsize=10)
    # ax.view_init(30, angle)

    ax.set_axis_off()
    ax.set_zlim(-4, 3) #control top whitespace
    ax.set_ylim(-2.5, 4) #control bottom whitespace
    fig.tight_layout()

    if save:
        plt.savefig('../ablations/fig1_graph.pdf', bbox_inches='tight')
        # plt.savefig('../ablations/fig1_graph.svg')

    if plot:
        fig.show()
    return fig, ax

if __name__ == "__main__":
    # plot_electric(plot=True, save=True)
    plot_greed(plot=True, save=True)
    # d3_ax()
    # quiver()
    # arrow()
    # draw0([4, 4])
    # draw([4, 2], [2, 4])
    # d3_quiver()
    # d3_arrow()