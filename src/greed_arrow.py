import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import sys
import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

from pylab import rcParams
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_geometric.data import Data
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
####################################################
# This part is just for reference if
# you are interested where the data is
# coming from
# The plot is at the bottom
#####################################################

def pca():
    # Generate some example data
    mu_vec1 = np.array([0,0,0])
    cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)

    mu_vec2 = np.array([1,1,1])
    cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

    # concatenate data for PCA
    samples = np.concatenate((class1_sample, class2_sample), axis=0)

    # mean values
    mean_x = mean(samples[:,0])
    mean_y = mean(samples[:,1])
    mean_z = mean(samples[:,2])

    #eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(cov_mat1)

    ################################
    #plotting eigenvectors
    ################################

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(samples[:,0], samples[:,1], samples[:,2], 'o', markersize=10, color='g', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
    for v in eig_vec:
        #ax.plot([mean_x,v[0]], [mean_y,v[1]], [mean_z,v[2]], color='red', alpha=0.8, lw=3)
        #I will replace this line with:
        a = Arrow3D([mean_x, v[0]], [mean_y, v[1]],
                    [mean_z, v[2]], mutation_scale=20,
                    lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')

    plt.title('Eigenvectors')

    plt.draw()
    plt.show()


#adapted from https://scipython.com/blog/visualizing-a-vector-field-with-matplotlib/
def E(q, r0, x, y):
    """Return the electric field vector E=(Ex,Ey) due to charge q at r0."""
    den = np.hypot(x-r0[0], y-r0[1])**3
    return q * (x - r0[0]) / den, q * (y - r0[1]) / den

def plot_electric(fig=None, ax=None, ax_idx=None, plot=False, save=False):
    # Grid of x, y points
    nx, ny = 64, 64
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Create a multipole with nq charges of alternating sign, equally spaced
    # on the unit circle.
    # nq = 2**int(sys.argv[1])
    nq = 2**int(1)#3)

    charges = []
    for i in range(nq):
        q = i%2 * 2 - 1
        charges.append((q, (np.cos(2*np.pi*i/nq), np.sin(2*np.pi*i/nq))))

    # Electric field vector, E=(Ex, Ey), as separate components
    Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
    for charge in charges:
        ex, ey = E(*charge, x=X, y=Y)
        Ex += ex
        Ey += ey

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    if ax is None:
        fig, ax = plt.subplots()
        ax_idx = 0

    # Plot the streamlines with an appropriate colormap and arrow style
    color = 2 * np.log(np.hypot(Ex, Ey))
    ax.streamplot(x, y, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)

    # Add filled circles for the charges themselves
    charge_colors = {True: '#aa0000', False: '#0000aa'}
    for q, pos in charges:
        ax.add_artist(Circle(pos, 0.05, color=charge_colors[q>0]))

    # add graph
    fig, ax = plot_greed_old(fig, ax)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_aspect('equal')
    if save:
        # plt.savefig('../ablations/figure1.pdf')
        plt.savefig('../ablations/figure1_a.pdf')
    if plot:
        fig.show()
    return fig, ax

def plot_greed_old(fig=None, ax=None, ax_idx=None, plot=False, save=False):
    edge_index = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4],
                               [1, 2, 3, 4, 0, 0, 0, 0, 2, 1, 4, 3]], dtype=torch.long)
    x = torch.tensor([[0, 0], [-1, -0.5], [-1, 0.5], [1, 0.5], [1, -0.5]], dtype=torch.float)

    color_vals = [x[i,0] for i in range(x.shape[0])]
    low, *_, high = sorted(color_vals)
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

    pos = x
    pos_i_dict = {i: [pos[i,0],pos[i,1]] for i in range(pos.shape[0])}
    graph = Data(x=x, edge_index=edge_index, pos=pos)

    # draw initial graph
    NXgraph = to_networkx(graph)
    if ax is None:
        fig, ax = plt.subplots()
        ax_idx = 0

    nx.draw(NXgraph, pos=pos_i_dict, ax=ax, node_size=200, #node_color=color,
            # cmap=plt.get_cmap('Spectral'),
            node_color=[mapper.to_rgba(i)
                        for i in color_vals],
            # with_labels=True,
            # font_color='white',
            arrows=False, width=0.25)

    limits = plt.axis('on')
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xticks([])
    ax.set_yticks([])
    if save:
        plt.savefig('../ablations/fig1_graph.pdf')
    if plot:
        fig.show()
    return fig, ax

def d3_ax():
    pass
def quiver():
    fig, ax = plt.subplots()
    x_pos = [0, -0.5]
    y_pos = [0, 0.5]
    x_direct = [1, 0]
    y_direct = [1, -1]
    ax.quiver(x_pos, y_pos, x_direct, y_direct, scale=5)
    ax.axis([-1.5, 1.5, -1.5, 1.5])
    plt.show()

def draw0(vec1):
    array = np.array([[0, 0, vec1[0], vec1[1]]])
    X, Y, U, V = zip(*array)
    plt.figure()
    plt.ylabel('Y-axis')
    plt.xlabel('X-axis')
    ax = plt.gca()
    ax.quiver(X, Y, U, V, color='b', angles='xy', scale_units='xy', scale=1)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    plt.draw()
    plt.show()

def arrow():
    hw = 0.02
    hl = 0.04

    plt.figure()
    plt.arrow(0.05, 0.15, 0.4, 0.15, head_width=hw, head_length=hl, length_includes_head=True)
    plt.text(0.2, 0.14, 'Vector A')
    plt.arrow(0.45, 0.3, 0.2, 0.5, head_width=hw, head_length=hl, length_includes_head=True)
    plt.text(0.6, 0.6, 'Vector B')
    plt.arrow(0.05, 0.15, 0.59, 0.65, head_width=hw, head_length=hl, length_includes_head=True)
    plt.text(0.28, 0.45, 'Vector A + Vector B', rotation=40)
    plt.title('Triangle Law of Vector Addition')
    plt.show()

def draw(x, y):
    xPlusy = (abs(x[0]+y[0]),abs(x[1]+y[1]))
    array = np.array([[0, 0, x[0], x[1]],
                      [x[0], x[1], y[0], y[1]],
                      [0, 0, xPlusy[0], xPlusy[1]]])
    print(array)
    X, Y, U, V = zip(*array)
    print("X =",X)
    print("Y =",Y)
    print("U =",U)
    print("V =",V)
    plt.figure()
    plt.ylabel('Y-axis')
    plt.xlabel('X-axis')
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy',color=['r','b','g'],scale=1)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    plt.draw()
    plt.show()

def d3_quiver():
    ax = plt.figure().add_subplot(projection='3d')
    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.8))
    # Make the direction data for the arrows
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))
    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
    plt.show()

def d3_arrow():
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    xs1 = [40,50,20]#34]
    ys1 = [30,30,30]
    zs1 = [98,46,70]#63]
    ax.scatter(xs1, ys1, zs1, s=100, c='g', marker='o')
    ax.text(xs1[0], ys1[0], zs1[0], '(%s,%s,%s)' % (str(xs1[0]), str(ys1[0]), str(zs1[0])), size=30, zorder=5, color='k')
    ax.text(xs1[1], ys1[1], zs1[1], '(%s,%s,%s)' % (str(xs1[1]), str(ys1[1]), str(zs1[1])), size=30, zorder=5, color='k')
    ax.text(xs1[2], ys1[2], zs1[2], '(%s,%s,%s)' % (str(xs1[2]), str(ys1[2]), str(zs1[2])), size=30, zorder=5, color='k')
    arw = Arrow3D([xs1[0],xs1[2]],[ys1[0],ys1[2]],[zs1[0],zs1[2]], arrowstyle="->", color="purple", lw = 5, mutation_scale=25)
    ax.add_artist(arw)
    ax.set_xlabel('X', fontsize=30, labelpad=20)
    ax.set_ylabel('Y', fontsize=30, labelpad=20)
    ax.set_zlabel('Z', fontsize=30, labelpad=20)
    plt.show()
