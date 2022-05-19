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
#       for lengthening axis
#     pass

def get_graph(graph_type):
    if graph_type == "square":
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                                   [1, 3, 0, 2, 1, 3, 0, 2]], dtype=torch.long)
        x = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=torch.float)
        y = torch.tensor([0, 1, 0, 1])
        clist = ["red", "green", "orange", "blue"]
    elif graph_type == "rectangle":
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                                   [1, 3, 0, 2, 1, 3, 0, 2]], dtype=torch.long)
        x_min, x_max = -2., 2.
        y_min, y_max = -0.6, 0.6
        x = torch.tensor([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]], dtype=torch.float)
        y = torch.tensor([0, 1, 0, 1])
        clist = 4 * ["black"]
    elif graph_type == "rectangle2":
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                                   [1, 3, 0, 2, 1, 3, 0, 2]], dtype=torch.long)
        x_min, x_max = -3.4, 3.4
        y_min, y_max = -0.6, 0.6
        x = torch.tensor([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]], dtype=torch.float)
        y = torch.tensor([0, 1, 0, 1])
        clist = 4 * ["black"]
    elif graph_type == "trapezium":
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],  # , 0, 2],
                                   [1, 3, 0, 2, 1, 3, 0, 2]],  # , 2, 0]],
                                  dtype=torch.long)
        x = torch.tensor([[-1, -1], [-1, 1], [1.5, 2], [1, -1]], dtype=torch.float)
        y = torch.tensor([0, 1, 0, 1])
        clist = ["red", "green", "orange", "blue"]
    elif graph_type == "barbell":
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5],
                                   [1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4]],
                                  dtype=torch.long)
        x = torch.tensor([[-1, -1], [-1, 1], [-0.5, 0], [0.5, 0], [1, 1], [1, -1]], dtype=torch.float)
        y = torch.tensor([0, 0, 1, 0, 1, 1])
        clist = ["red", "red", "blue", "red", "blue", "blue"]
        # clist = ["red", "green", "orange", "blue", "purple", "c"]
    elif graph_type == "barbell2":
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 0, 1, 4, 5],
                                   [1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4, 6, 6, 7, 7]],
                                  dtype=torch.long)
        # x = torch.tensor([[-1, -1], [-1, 1], [-0.5, 0], [0.5, 0], [1, 1], [1, -1], [-1.25, 0], [1.25, 0]], dtype=torch.float)
        h = 0.5
        x = torch.tensor([[-1, -h], [-1, h], [-0.5, 0], [0.5, 0], [1, h], [1, -h], [-1.25, 0], [1.25, 0]], dtype=torch.float)
        y = torch.tensor([0, 0, 0, 1, 0, 1])
        clist = ["red", "red", "blue", "red", "blue", "blue", "blue", "red"]

    return edge_index, x, y, clist

def get_data(edge_index, x, y):
    num_nodes = x.shape[0]
    num_classes = y.max()
    data = DummyData(edge_index, None, num_nodes, x, y, num_classes)
    eig_vecs = torch.tensor([[1, 0],[0, 1]], dtype=torch.float)
    # eig_vecs = torch.tensor([[0, 1],[1, 0]], dtype=torch.float)
    # invsqrt2 = 1 / torch.sqrt(torch.tensor(2))
    # eig_vecs = invsqrt2 * torch.tensor([[1, 1],[1, -1]], dtype=torch.float)
    eig_vals = torch.tensor([-2.0, 2.0], dtype=torch.float)
    W = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
    T = 2.8#2.5
    dt = 0.1
    data.W = W
    data.T = T
    data.dt = dt
    return data

def get_eig_pos(data, evec_list):
    A = get_adj(data.edge_index)
    A_eval, A_evec, W_eval, W_evec, WA_eval, WA_evec = get_eigs(A, data.W)
    xev = torch.zeros(data.x.shape)
    for i, e in enumerate(evec_list):
        for n in range(data.num_nodes):
            xev[n,0] += WA_evec[n,e]
            xev[n,1] += WA_evec[data.num_nodes+n,e]
    return xev

def get_adj(edge_index):
    adj = to_dense_adj(edge_index).squeeze()
    deginvsq = torch.pow(torch.sqrt(adj.sum(dim=0)), -1)
    A = torch.diag(deginvsq) @ adj @ torch.diag(deginvsq)
    return A

def get_eigs(A,W):
    A_eval, A_evec = torch.linalg.eigh(A)
    W_eval, W_evec = torch.linalg.eigh(W)
    WA = torch.kron(W, A)
    WA_eval, WA_evec = torch.linalg.eigh(WA)
    print(f"A {A}")
    print(f"A_eval {A_eval}")
    print(f"A_evec (per row) {A_evec.T}")
    print(f"WA {WA}")
    print(f"WA_evec {WA_evec}")
    print(f"WA_eval (per row) {WA_eval.T}")
    print(f"lambda_+ {W_evec.max()}")
    print(f"lambda_- {W_evec.min()}")
    L = torch.eye(A.shape[0]) - A
    L_eval, L_evec = torch.linalg.eigh(L)
    print(f"L_eval {L_eval}")
    print(f"L_evec (per row) {L_evec.T}")
    row_L = torch.abs(L_eval).max()
    print(f"row_L_ {row_L}")
    print(f"row_minus {torch.abs(W_evec.min())*(1-row_L)}")
    return A_eval, A_evec, W_eval, W_evec, WA_eval, WA_evec

def get_dynamics(data):
    W, T, dt = data.W, data.T, data.dt
    edge_index = to_undirected(data.edge_index)
    A = get_adj(edge_index)
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


def plot_WA_eigs(data, offset, t_idx, X_all, evec_list, escale, clist, ax):
    edge_index = data.edge_index
    W = data.W
    A = get_adj(edge_index)
    A_eval, A_evec, W_eval, W_evec, WA_eval, WA_evec = get_eigs(A, W)
    N = X_all.shape[0]
    WA_normed = WA_evec / WA_evec.norm(p=2, dim=0).unsqueeze(0) * escale
    WA_normed = WA_normed.detach().numpy()
    for i, e in enumerate(evec_list):
        for n in range(N):
            xpos = X_all[n, 0, t_idx]
            x1pos = xpos + WA_normed[n,e]
            ypos = X_all[n, 1, t_idx]
            y1pos = ypos + WA_normed[N+n,e]
            arw = Arrow3D([offset,offset],[xpos,x1pos],[ypos,y1pos], arrowstyle="->", color=clist[i], lw = 2, mutation_scale=25)
            ax.add_artist(arw)

def plot_attr_rep(data, x, y, t, xscale, yscale, clist, label_pos, ax):
    edge_index = data.edge_index
    W = data.W
    A = get_adj(edge_index)
    A_eval, A_evec, W_eval, W_evec, WA_eval, WA_evec = get_eigs(A, W)
    labels = ["rep","attr"]
    for n in range(2):
        xpos = x
        x1pos = x + W_evec[0,n] * xscale
        ypos = y
        y1pos = y + W_evec[1,n] * yscale
        arw = Arrow3D([t, t], [xpos, x1pos], [ypos, y1pos], arrowstyle="->", color=clist[n], lw=2,
                      mutation_scale=25)
        ax.add_artist(arw)
        label_scale = 1.3
        ax.text(t, label_pos[n][0], label_pos[n][1], labels[n], c=clist[n], size=12)

def plot_time(x, y, t, T, label_pos, color, ax, lw=1.):
    xpos = x
    ypos = y
    arw = Arrow3D([t, T], [xpos, xpos], [ypos, ypos], arrowstyle="->", color=color, lw=lw,
                  mutation_scale=25)
    ax.add_artist(arw)
    ax.text(label_pos[2], label_pos[0], label_pos[1], "time", c=color, size=12)


def plot_labels(labels, offset, X_all, clist, t_idx, T, ax):
    N = X_all.shape[0]
    num_T = X_all.shape[-1]
    #plot initial labels
    for n in range(N):
        ax.text(T, X_all[n,0,t_idx]*offset[n], X_all[n,1,t_idx]*offset[n], f"{labels[n]}", c=clist[n], size=14)

def plot_slices(data, X_all, Y_all, num_slices, clist, edge_col, ax, nodes=True, edges=True, arrows=True, trace=False):
    edge_index = data.edge_index.detach().numpy()
    N = X_all.shape[0]
    T = X_all.shape[-1]
    t_idxs = [0] + list(range(T//num_slices, T - T//num_slices, T//num_slices)) + [T-1]
    dt = data.dt
    #plot slices
    for t in t_idxs:
        pos = {i: [t*dt, X_all[i, 0, t].item(), X_all[i, 1, t].item()]
               for i in range(N)}
        node_xyz = np.array([pos[i] for i in range(N)])
        edge_xyz = np.array([(pos[uv[0]], pos[uv[1]]) for uv in edge_index.T])
        plot_3d(node_xyz, edge_xyz, clist, edge_col, ax, nodes, edges)

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
                pass #terminal arrows
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

# def plot_graph(data, X_all, t_idx, clist, ax, nodes=True, edges=True):
#     edge_index = data.edge_index.detach().numpy()
#     N = X_all.shape[0]
#     dt = data.dt
#     pos = {i: [t_idx * dt, X_all[i, 0, t_idx].item(), X_all[i, 1, t_idx].item()]
#            for i in range(N)}
#     node_xyz = np.array([pos[i] for i in range(N)])
#     edge_xyz = np.array([(pos[uv[0]], pos[uv[1]]) for uv in edge_index.T])
#     plot_3d(node_xyz, edge_xyz, clist, ax, nodes, edges)

def plot_graph(data, X, t, clist, edge_col, ax, nodes=True, edges=True, lw=1):
    edge_index = data.edge_index.detach().numpy()
    N = X.shape[0]
    pos = {i: [t, X[i, 0], X[i, 1]] for i in range(N)}
    node_xyz = np.array([pos[i] for i in range(N)])
    edge_xyz = np.array([(pos[uv[0]], pos[uv[1]]) for uv in edge_index.T])
    plot_3d(node_xyz, edge_xyz, clist, edge_col, ax, nodes, edges, lw=lw)

def plot_arrows(X_all, t, t_idx, dt_idx, colors, ax):
    N = X_all.shape[0]
    for n in range(N):
        xpos = X_all[n, 0, t_idx]
        x1pos = X_all[n, 0, t_idx + dt_idx]
        ypos = X_all[n, 1, t_idx]
        y1pos = X_all[n, 1, t_idx + dt_idx]
        arw = Arrow3D([t, t], [xpos, x1pos], [ypos, y1pos], arrowstyle="->", color=colors[n], lw=2,
                      mutation_scale=25)
        ax.add_artist(arw)

def plot_trace_dots(X_all, T_idx, dt, colors, ax):
    N = X_all.shape[0]
    for n in range(N):
        ax.scatter(dt * np.linspace(0, T_idx, T_idx+1), X_all[n, 0, :], X_all[n, 1, :],
                   s=10.0, color=colors[n], ec="w")

def plot_trace_lines(X_all, T_idx, dt, colors, ax):
    N = X_all.shape[0]
    for n in range(N):
        ax.plot(dt * np.linspace(0, T_idx, T_idx+1), X_all[n, 0, :], X_all[n, 1, :],
                   lw=2.0, color=colors[n])

# https://networkx.org/documentation/stable/auto_examples/3d_drawing/plot_basic.html
def plot_3d(node_xyz, edge_xyz, clist, edge_col, ax, nodes, edges, lw=1):
    if nodes:
        # Plot the nodes - alpha is scaled by "depth" automatically
        # cmap = plt.get_cmap("tab10")
        ax.scatter(*node_xyz.T, c=clist, s=80)#cmap=cmap#, ec="w")
    if edges:
        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color=edge_col, lw=lw)
            # ax.plot(*vizedge.T, color="black", lw=1)

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

def plot_frames(data, t_idxs, ax, scale):
    X_all, Y_all = get_dynamics(data)
    clist= 4*["grey"]
    for t_idx in t_idxs:
        t = t_idx * data.dt
        X = scale * X_all[:,:,t_idx]
        plot_graph(data, X, t, clist, ax, nodes=False, edges=True)
    # plot_slices(data, X_all, Y_all, num_slices, clist, ax, nodes=False, edges=True, arrows=False)

def plot_greed(fig=None, ax=None, ax_idx=None, plot=False, save=False):
    if ax is None:
        fig = plt.figure()#figsize=(12,8))
        ax = fig.add_subplot(111, projection="3d")

    #plot frames
    edge_index, x, y, clist = get_graph("square")
    data = get_data(edge_index, x, y)
    num_slices = 1#3 #excluding initial condition
    T_idx = int(np.round(data.T/data.dt, 2)) #int(data.T/data.dt)
    step = int(T_idx//num_slices)
    t_idxs = list(range(0, T_idx + step, step))
    scale = 2.5
    # plot_frames(data, t_idxs, ax, scale)

    #plot manual frame
    edge_index, x, y, clist = get_graph("rectangle")
    data = get_data(edge_index, x, y)
    plot_graph(data, x, 0, clist, "black", ax, nodes=False, edges=True, lw=0.75)
    edge_index, x, y, clist = get_graph("rectangle2")
    data = get_data(edge_index, x, y)
    plot_graph(data, x, 2.8, clist, "black", ax, nodes=False, edges=True, lw=0.75)

    #plot graph
    # edge_index, x, y, clist = get_graph("trapezium")
    # edge_index, x, y, clist = get_graph("barbell")
    edge_index, x, y, clist = get_graph("barbell2")
    data = get_data(edge_index, x, y)
    evec_list = [0,1,2, -3, -2, -1] #goes most neg to most pos of WA
    # data.x = get_eig_pos(data, evec_list)

    X_all, Y_all = get_dynamics(data)

    #plot_labels
    # offset = [1.8,2.2,1.2,1.9] #for square
    # offset = num_nodes * [1.2]
    # plot_labels(data.y, offset, X_all, num_nodes * ["black"], 0, 0, ax)

    #plot trace
    plot_trace_dots(X_all, T_idx, data.dt, clist, ax)
    # plot_trace_lines(X_all, T_idx, data.dt, clist, ax)

    #plot T=0
    # plot_slices(data, X_all, Y_all, num_slices, clist, ax, nodes=True, edges=False, arrows=True, trace=True)
    plot_graph(data, X_all[:,:,0], 0, clist, "tab:gray", ax, nodes=True, edges=True)

    #plot t=T
    plot_graph(data, X_all[:,:,T_idx], data.T, clist, "tab:grey", ax, nodes=True, edges=True)

    #plot eigs arrows
    clist = ["red", "blue"] # clist = ["tab:blue", "tab:purple"]
    evec_list = [-1, 0]
    escale = 3
    t_idx = 0
    # plot_WA_eigs(data, 0, t_idx, X_all, evec_list, escale, clist, ax)

    #plot attraction repulsion axis
    y_shift = -0.25
    label_pos = [[3.9, 0.2 + y_shift], [3.0, 0.65 + y_shift]]
    # plot_attr_rep(data, x=3., y=0.+y_shift, t=1.0, xscale=1.5, yscale=0.6, clist=2*["tab:grey"], label_pos=label_pos, ax=ax)

    # x, y = 0, 0
    x, y = -2., -0.6
    plot_time(x, y, t=0, T=2.8, label_pos=[x-0.2, y-0.2, 2.9], color="black", ax=ax, lw=0.75)

    #foramt axis
    # ax.set_xlabel('time', fontsize=10)
    # ax.set_ylabel('x0-axis', fontsize=10)
    # ax.set_zlabel('x1-axis', fontsize=10)
    # ax.view_init(30, angle)

    ax.set_axis_off()
    ax.set_zlim(-1, 0.2) #control top whitespace
    ax.set_ylim(-1.26, 3) #control bottom whitespace
    ax.set_xlim(0.0, 2.8) #control time axis left/right whitespace

    fig.tight_layout()

    if save:
        plt.savefig('../ablations/fig1_graph.pdf', bbox_inches='tight')
        # plt.savefig('../ablations/fig1_graph.svg')

    if plot:
        fig.show()
    return fig, ax

if __name__ == "__main__":
    plot_greed(plot=True, save=True)

    #scenarios tried
    # trapezoid
    # barbell
    #plot in most attractive/repulsive eigen vectors
    #plot dominant repulsive eigen vector at t=0

    #problem is that node coordinates are the features so can't really show a mixed up graph