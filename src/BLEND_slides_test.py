import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation


def simple_update(num, n, layout, G, ax):
    ax.clear()

    # Draw the graph with random node colors
    random_colors = np.random.randint(2, size=n)
    nx.draw(G, pos=layout, node_color=random_colors, ax=ax)

    # Set the title
    ax.set_title("Frame {}".format(num))


def simple_animation():

    # Build plot
    fig, ax = plt.subplots(figsize=(6,4))

    # Create a graph and layout
    n = 30 # Number of nodes
    m = 70 # Number of edges
    G = nx.gnm_random_graph(n, m)
    layout = nx.spring_layout(G)

    ani = animation.FuncAnimation(fig, simple_update, frames=10, fargs=(n, layout, G, ax))
    ani.save('../BLEND_animation/animation_1.gif', writer='imagemagick')

    plt.show()

simple_animation()