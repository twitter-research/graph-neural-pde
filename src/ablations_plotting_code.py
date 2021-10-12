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
    plt.show()


# Euclidean and hyperbolic positional embeddings
def fig2():
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





#Step size ablation
def fig3():
    pass

if __name__ == "__main__":
    # fig1a()
    # fig1b()
    fig2()