import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

#With/without positional coordinates ablation
def fig1a():
    datasets = ['Cora','CiteSeer','PubMed','Coauthor','CS Computer','Photo']
    x = np.arange(len(datasets))

    BLEND_mean = [84.8, 75.9, 79.5, 92.9, 86.9, 92.9]
    BLEND_sd = [0.9, 1.3, 1.4, 0.2, 0.6, 0.6]

    GRAND_mean = [83.6, 73.4, 78.8, 92.9, 83.7, 92.3]
    GRAND_sd = [1.0, 0.5, 1.7, 0.4, 1.2, 0.9]

    sns.set()
    sns.set_style("ticks")

    fig, ax = plt.subplots()

    width = 0.45
    rects1 = ax.bar(x - width/2, BLEND_mean, width=width, yerr=BLEND_sd, label='BLEND')
    rects2 = ax.bar(x + width/2, GRAND_mean, width=width, yerr=GRAND_sd, label='BLEND w/o positional')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracy')
    ax.grid(axis='y')
    # ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    # ax.xticks(datasets, rotation=45, fontsize=9)
    ax.set_ylim([70, 100])
    ax.legend(loc='upper left', ncol=2)

    ax.bar_label(rects1, padding=3, fontsize=11, color=sns.color_palette()[0])
    ax.bar_label(rects2, padding=3, fontsize=11, color=sns.color_palette()[1])

    fig.tight_layout()
    plt.show()

#With/without positional coordinates and GAT
def fig1b():
    datasets = ['Cora','CiteSeer','PubMed','Coauthor','CS Computer','Photo']
    x = 1.6 * np.arange(len(datasets))

    BLEND_mean = [84.8, 75.9, 79.5, 92.9, 86.9, 92.9]
    BLEND_sd = [0.9, 1.3, 1.4, 0.2, 0.6, 0.6]

    GRAND_mean = [83.6, 73.4, 78.8, 92.9, 83.7, 92.3]
    GRAND_sd = [1.0, 0.5, 1.7, 0.4, 1.2, 0.9]

    # GATwPOS_mean = [82.59, 74.54, 76.82, 90.32, 73.94, 76.72]
    GATwPOS_mean = [82.6, 74.5, 76.8, 90.3, 74.0, 76.7]
    GATwPOS_sd = [1.2, 1.9, 1.6, 0.3, 23.5, 29.1]

    sns.set()
    sns.set_style("ticks")

    fig, ax = plt.subplots()

    width = 0.5
    rects1 = ax.bar(x - width, BLEND_mean, width=width, yerr=BLEND_sd, label='BLEND')
    rects2 = ax.bar(x, GRAND_mean, width=width, yerr=GRAND_sd, label='BLEND w/o positional')
    rects3 = ax.bar(x + width, GATwPOS_mean, width=width, yerr=GATwPOS_sd, label='GAT w. POS')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracy')
    ax.grid(axis='y')

    # ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    # ax.xticks(datasets, rotation=45, fontsize=9)
    ax.set_ylim([70, 100])
    ax.legend(loc='upper left', ncol=2)

    fs = 9
    ax.bar_label(rects1, padding=6, fontsize=fs, color=sns.color_palette()[0])
    ax.bar_label(rects2, padding=3, fontsize=fs, color=sns.color_palette()[1])
    ax.bar_label(rects3, padding=0, fontsize=fs, color=sns.color_palette()[2])

    fig.tight_layout()
    plt.savefig('../ablations/Neurips_plots/w_wo_pos_encs_abl.pdf')
    plt.show()


# Euclidean and hyperbolic positional embeddings
def fig2():
    # LHS
    datasets = ['Cora','CiteSeer','PubMed','Coauthor','CS Computer','Photo']
    x = np.arange(len(datasets))

    BLEND_mean = [84.8, 75.9, 79.5, 92.9, 86.9, 92.9]
    BLEND_sd = [0.9, 1.3, 1.4, 0.2, 0.6, 0.6]

    Hyp_mean = [84.2, 75.9, 79.6, 92.5, 86.4, 93.1]
    Hyp_sd = [0.9, 1.2, 0.4, 0.2, 0.7, 0.2]
    sns.set()
    sns.set_style("ticks")

    fig, ax = plt.subplots()

    width = 0.45
    rects1 = ax.bar(x - width/2, BLEND_mean, width=width, yerr=BLEND_sd, label='BLEND')
    rects2 = ax.bar(x + width/2, Hyp_mean, width=width, yerr=Hyp_sd, label='Hyp')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracy')
    ax.grid(axis='y')
    # ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    # ax.xticks(datasets, rotation=45, fontsize=9)
    ax.set_ylim([70, 100])
    ax.legend(loc='upper left', ncol=2)

    ax.bar_label(rects1, padding=3, fontsize=11, color=sns.color_palette()[0])
    ax.bar_label(rects2, padding=3, fontsize=11, color=sns.color_palette()[1])

    fig.tight_layout()
    plt.savefig('../ablations/Neurips_plots/blend_hyp.pdf')
    plt.show()
    #RHS - might not be needed hyp plot
    #build_ablation_fig.ipynb
    #'../ablations/Neurips_plots/pos_emb_dim.pdf'

#Step size ablation
def fig3():

    x_labels = ['Dopri5', '0.25', '0.5', '1', '2', '4', '8']
    x = np.arange(len(x_labels))

    mean = {
    'Cora':	[83.8, 84.3, 83.4, 83.5, 84.3, 40.9, 35.1],
    'Citeseer':	[74.3, 73.1, 74.2, 73.3, 74.9, 59.9, 23.3],
    'Pubmed':	[80.0, 78.8, 79.8, 79.2, 78.6, 72.4, 46.8],
    'CoauthorCS':	[92.8, 92.8, 92.7, 92.5, 91.6, 77.1, 77.1],
    'Computers':	[85.1, 84.8, 85.0, 85.0, 84.8, 75.8, 75.2],
    'Photo':	[92.8, 92.9, 92.6, 92.3, 91.7, 82.7, 83.2]}

    std = {
    'Cora':	[1.3, 0.8, 1.1, 1.7, 1.1, 3.1, 2.8],
    'Citeseer':	[2.3, 1.8, 1.6, 2.2, 1.1, 3.8, 3.2],
    'Pubmed':	[1.1, 1.4, 1.5, 1.3, 1.9, 2.5, 1.9],
    'CoauthorCS':	[0.1, 0.2, 0.3, 0.1, 0.1, 0.3, 0.4],
    'Computers':	[0.5, 0.6, 0.6, 0.8, 0.4, 1.1, 0.8],
    'Photo':	[0.4, 0.3, 0.2, 0.3, 0.5, 0.8, 0.8]}

    sns.set()
    sns.set_style("ticks")

    fig, (ax, ax2) = plt.subplots(1, 2, sharex=True)

    # plot the same data on both axes
    for i, (k, v) in enumerate(mean.items()):
        ax.plot(x, v, label=k)
        ax2.plot(x, v, label=k)

    # ax.set_xticks(x)
    ax.set_xticklabels(x_labels)


    ax.set_xlim(0, 0.25)
    ax2.set_xlim(0.75, 8)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright=False)
    ax2.tick_params(bottom=True, top=False, left=False, right=False)
    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    ax2.yaxis.tick_right()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    plt.show()


def fig4():

    x_labels = ['Dopri5', '0.25', '0.5', '1', '2', '4', '8']
    x = np.arange(len(x_labels))

    mean = {
    'Cora':	[83.8, 84.3, 83.4, 83.5, 84.3, 40.9, 35.1],
    'Citeseer':	[74.3, 73.1, 74.2, 73.3, 74.9, 59.9, 23.3],
    'Pubmed':	[80.0, 78.8, 79.8, 79.2, 78.6, 72.4, 46.8],
    'CoauthorCS':	[92.8, 92.8, 92.7, 92.5, 91.6, 77.1, 77.1],
    'Computers':	[85.1, 84.8, 85.0, 85.0, 84.8, 75.8, 75.2],
    'Photo':	[92.8, 92.9, 92.6, 92.3, 91.7, 82.7, 83.2]}

    std = {
    'Cora':	[1.3, 0.8, 1.1, 1.7, 1.1, 3.1, 2.8],
    'Citeseer':	[2.3, 1.8, 1.6, 2.2, 1.1, 3.8, 3.2],
    'Pubmed':	[1.1, 1.4, 1.5, 1.3, 1.9, 2.5, 1.9],
    'CoauthorCS':	[0.1, 0.2, 0.3, 0.1, 0.1, 0.3, 0.4],
    'Computers':	[0.5, 0.6, 0.6, 0.8, 0.4, 1.1, 0.8],
    'Photo':	[0.4, 0.3, 0.2, 0.3, 0.5, 0.8, 0.8]}

    sns.set()
    sns.set_style("ticks")

    fig, ax = plt.subplots()

    # plot the same data on both axes
    for i, (k, v) in enumerate(mean.items()):
        ax.plot(x, v, label=k)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='best', ncol=2)

    ax.set_xlabel('Stepsize')
    ax.set_ylabel('Accuracy')

    plt.show()


def fig5():
    from brokenaxes import brokenaxes

    x_labels = ['Dopri5', '0.25', '0.5', '1', '2', '4', '8']
    x = np.arange(len(x_labels))

    mean = {
        'Cora': [83.8, 84.3, 83.4, 83.5, 84.3, 40.9, 35.1],
        'Citeseer': [74.3, 73.1, 74.2, 73.3, 74.9, 59.9, 23.3],
        'Pubmed': [80.0, 78.8, 79.8, 79.2, 78.6, 72.4, 46.8],
        'CoauthorCS': [92.8, 92.8, 92.7, 92.5, 91.6, 77.1, 77.1],
        'Computers': [85.1, 84.8, 85.0, 85.0, 84.8, 75.8, 75.2],
        'Photo': [92.8, 92.9, 92.6, 92.3, 91.7, 82.7, 83.2]}

    std = {
        'Cora': [1.3, 0.8, 1.1, 1.7, 1.1, 3.1, 2.8],
        'Citeseer': [2.3, 1.8, 1.6, 2.2, 1.1, 3.8, 3.2],
        'Pubmed': [1.1, 1.4, 1.5, 1.3, 1.9, 2.5, 1.9],
        'CoauthorCS': [0.1, 0.2, 0.3, 0.1, 0.1, 0.3, 0.4],
        'Computers': [0.5, 0.6, 0.6, 0.8, 0.4, 1.1, 0.8],
        'Photo': [0.4, 0.3, 0.2, 0.3, 0.5, 0.8, 0.8]}

    sns.set()
    sns.set_style("ticks")

    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(6, 4))
    baxes = brokenaxes(xlims=((0, 0.25), (0.25, 7)), hspace=.05)

    # plot the same data on both axes
    for i, (k, v) in enumerate(mean.items()):
        baxes.plot(x, v, label=k)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='best', ncol=2)

    ax.set_xlabel('Stepsize')
    ax.set_ylabel('Accuracy')

    plt.show()

def fig6():

    x_labels = ['Dopri5', '0.25', '0.5', '1', '2', '4', '8']
    x = np.arange(len(x_labels))

    means = {
    'Cora':	[83.8, 84.3, 83.4, 83.5, 84.3, 40.9, 35.1],
    'Citeseer':	[74.3, 73.1, 74.2, 73.3, 74.9, 59.9, 23.3],
    'Pubmed':	[80.0, 78.8, 79.8, 79.2, 78.6, 72.4, 46.8],
    'CoauthorCS':	[92.8, 92.8, 92.7, 92.5, 91.6, 77.1, 77.1],
    'Computers':	[85.1, 84.8, 85.0, 85.0, 84.8, 75.8, 75.2],
    'Photo':	[92.8, 92.9, 92.6, 92.3, 91.7, 82.7, 83.2]}

    stds = {
    'Cora':	[1.3, 0.8, 1.1, 1.7, 1.1, 3.1, 2.8],
    'Citeseer':	[2.3, 1.8, 1.6, 2.2, 1.1, 3.8, 3.2],
    'Pubmed':	[1.1, 1.4, 1.5, 1.3, 1.9, 2.5, 1.9],
    'CoauthorCS':	[0.1, 0.2, 0.3, 0.1, 0.1, 0.3, 0.4],
    'Computers':	[0.5, 0.6, 0.6, 0.8, 0.4, 1.1, 0.8],
    'Photo':	[0.4, 0.3, 0.2, 0.3, 0.5, 0.8, 0.8]}

    sns.set()
    sns.set_style("ticks")

    fig, ax = plt.subplots()

    # plot the same data on both axes
    for i, (k, v) in enumerate(means.items()):
        ax.scatter(x[0], v[0], marker='x')
        ax.plot(x[1:], np.array(v[1:]), label=k, c=sns.color_palette()[i])
        ax.plot(x[:2], np.array(v[:2]), c=sns.color_palette()[i], linestyle='--')

        # mean = np.array(v[1:])
        # std = np.array(stds[k][1:])
        # ax.fill_between(x[1:], mean - std, mean + std, alpha=0.3, facecolor=sns.color_palette()[i])

        mean = np.array(v)
        std = np.array(stds[k])
        ax.fill_between(x, mean - std, mean + std, alpha=0.3, facecolor=sns.color_palette()[i])


    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='best', ncol=2)
    ax.grid(axis='y')
    ax.set_xlabel('Stepsize')
    ax.set_ylabel('Accuracy')
    plt.savefig('../ablations/Neurips_plots/step_size_abl.pdf')

    plt.show()

if __name__ == "__main__":
    # fig1a() #With/without positional coordinates ablation
    fig1b() #With/without positional coordinates and GAT
    fig2() # Euclidean and hyperbolic positional embeddings
    # Step size ablation
    # fig3()
    # fig4()
    # fig5()
    fig6()

    # broken_X()

