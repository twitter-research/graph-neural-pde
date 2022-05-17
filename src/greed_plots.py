import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib. pyplot as plt
from seaborn.relational import _RelationalPlotter, _ScatterPlotter
from matplotlib.ticker import ScalarFormatter

sns.set_theme(style="whitegrid")
rs = np.random.RandomState(365)
values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()
sns.lineplot(data=data, palette="tab10", linewidth=2.5)
# plt.show()

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

f = plt.figure(figsize=(6, 6))
gs = f.add_gridspec(2, 2)
with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[0, 0])
    sinplot()
with sns.axes_style("white"):
    ax = f.add_subplot(gs[0, 1])
    sinplot()
with sns.axes_style("ticks"):
    ax = f.add_subplot(gs[1, 0])
    sinplot()
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[1, 1])
    sinplot()
f.tight_layout()
# plt.show()

def jitter():
    # Import Data
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
    # Draw Stripplot
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    sns.stripplot(df.cty, df.hwy, jitter=0.25, size=8, ax=ax, linewidth=.5)
    # Decorations
    plt.title('Use jittered plots to avoid overlapping of points', fontsize=22)
    # fig.show()

def size_d_plot(path, plot=True, save=True):
    df = pd.read_csv(path)
    df = df.replace(to_replace={"diag_dom":"diag-dom"})
    for ds in ["chameleon","squirrel","Cora"]:
        ds_df = df[(df.dataset == ds)]
        mask = (df["function"] == "greed_non_linear")
        ds_df.loc[mask, 'function'] = ds_df.loc[mask, 'gnl_W_style']

        piv = pd.pivot_table(ds_df, values="test_mean", index="hidden_dim", columns="function",
                                 aggfunc=np.max)
        fs = 16
        fig, ax = plt.subplots()
        sns.lineplot(data=piv, palette="tab10", linewidth=2.5, ax=ax)
        ax.set(xscale='log')
        ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())

        ax.set_ylabel('Test acc', fontsize=fs)
        ax.set_title(f"dataset {ds}", fontsize=fs)
        # ax.get_xaxis().set_visible(False)
        ax.legend(prop=dict(size=fs), loc='upper left')
        if save:
            plt.savefig(f"../ablations/size_d_{ds}.pdf")
        if plot:
            fig.show()
    #todo make multi-subplots

def syn_cora_plot(path, fig=None, ax=None, ax_idx=None, plot=False, save=False):
    df = pd.read_csv(path)
    df = df.replace(to_replace={"diag_dom":"diag-dom", "neg_prod":"neg-prod"})
    gnl_df = df[(df.function == "greed_non_linear")]
    gnl_piv = pd.pivot_table(gnl_df, values="test_mean", index="target_homoph", columns="gnl_W_style", aggfunc=np.max)
    base_df = df[(df.function != "greed_non_linear")]
    base_piv = pd.pivot_table(base_df, values="test_mean", index="target_homoph", columns="function", aggfunc=np.max)
    piv = pd.merge(gnl_piv, base_piv, on=['target_homoph'])

    fs = 16
    if ax is None:
        fig, ax = plt.subplots()
        sns.lineplot(data=piv, palette="tab10", linewidth=2.5, ax=ax)
        # ax.set_xlabel('Target homophily', fontsize=fs)
        ax.set_ylabel('Test acc', fontsize=fs)
        ax.get_xaxis().set_visible(False)
        ax.legend(prop=dict(size=fs), loc='upper left')
    else:
        sns.lineplot(data=piv, palette="tab10", linewidth=2.5, ax=ax[ax_idx])
        ax[ax_idx].set_xlabel('Target homophily', fontsize=fs)
        ax[ax_idx].set_ylabel('Test acc', fontsize=fs)
        ax[ax_idx].legend(prop=dict(size=fs), loc='upper left')
    if save:
        plt.savefig('../ablations/syn_cora_plot.pdf')
    if plot:
        fig.show()
    return fig, ax

def syn_cora_gcn_plot(path, fig=None, ax=None, ax_idx=None, plot=False, save=False):
    df = pd.read_csv(path)
    replace_dict = {0:"0:gcn",1:"1:gcn_enc/dec",2:"2:gcn_residual",3:"3:gcn_share_W",4:"4:gcn_symm_W",5:"5:no_nonLin"}
    df.loc[:,'gcn_params_idx'].replace(to_replace=replace_dict, inplace=True)

    piv = pd.pivot_table(df, values="test_mean", index="target_homoph", columns="gcn_params_idx", aggfunc=np.max)
    # base_df = df[(df.function != "greed_non_linear")]
    # base_piv = pd.pivot_table(base_df, values="test_mean", index="target_homoph", columns="function", aggfunc=np.max)
    # piv = pd.merge(gnl_piv, base_piv, on=['target_homoph'])
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots()
        sns.lineplot(data=piv, palette="tab10", linewidth=2.5, ax=ax)
        ax.set_xlabel('Target homophily', fontsize=14)
        ax.set_ylabel('Test acc', fontsize=14)
        ax.legend(prop=dict(size=12))
    else:
        sns.lineplot(data=piv, palette="tab10", linewidth=2.5, ax=ax[ax_idx])
        ax[ax_idx].set_xlabel('Target homophily', fontsize=14)
        ax[ax_idx].set_ylabel('Test acc', fontsize=14)
        ax[ax_idx].legend(prop=dict(size=12))

    if save:
        plt.savefig('../ablations/syn_cora_gcn_plot.pdf')
    if plot:
        fig.show()
    return fig, ax


def get_max_df(path):
    df = pd.read_csv(path)
    df = df[(df.function == "greed_non_linear")].reset_index(drop=True)
    max_idxs = []
    for target_homoph in ['0.00', '0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00']:
        for w_type in ["diag_dom", "neg_prod", "prod", "sum"]:
            new_cols = ["target_homoph", "gnl_W_style", "time", "step_size", "test_mean"]
            newdf = df[new_cols][(df.target_homoph == float(target_homoph)) & (df.gnl_W_style == w_type)]
            max_idx = newdf[['test_mean']].idxmax()
            max_idxs.append(int(max_idx.values))
            # max_time = df.iloc[max_idx]['time'] #.values
            # max_step = df.iloc[max_idx]['step_size'] #.values
    max_df = df.iloc[max_idxs]
    return max_df

def syn_cora_best_times(path):
    max_df = get_max_df(path)
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    sns.stripplot(x=max_df.target_homoph, y=max_df.time, hue=max_df.gnl_W_style, jitter=0.15, size=8, ax=ax, linewidth=.5)
    plt.title('Best times', fontsize=22)
    fig.show()
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    sns.stripplot(x=max_df.target_homoph, y=max_df.step_size, hue=max_df.gnl_W_style, jitter=0.15, size=8, ax=ax, linewidth=.5)
    plt.title('Best steps', fontsize=22)
    fig.show()

def syn_cora_energy(path, fig=None, ax=None, ax_idx=None):
    max_df = get_max_df(path)
    # max_df['W_homoph'] = max_df.apply(lambda row: str(row.gnl_W_style) + "_" + str(row.target_homoph), axis=1)

    # fig, ax = plt.subplots()
    # ax.set_title(f"T0_dirichlet_energy_mean", fontdict={'fontsize': 24})
    # sns.stripplot(x="target_homoph", y="T0_dirichlet_mean", hue="gnl_W_style", jitter=False, data=max_df, ax=ax)
    # fig.show()
    # fig, ax = plt.subplots()
    # ax.set_title(f"TN_dirichlet_energy_mean", fontdict={'fontsize': 24})
    # sns.stripplot(x="target_homoph", y="TN_dirichlet_mean", hue="gnl_W_style", jitter=0.15, data=max_df, ax=ax)
    # fig.show()
    # #combined
    # new_cols = ["target_homoph", "gnl_W_style", "T0_dirichlet_mean"]
    # df1 = max_df[new_cols]
    # df1 = df1.rename(columns={"T0_dirichlet_mean": "dirichlet_mean"})
    # df1['energy_time'] = 'T0'
    # new_cols = ["target_homoph", "gnl_W_style", "TN_dirichlet_mean"]
    # df2 = max_df[new_cols]
    # df2 = df2.rename(columns={"TN_dirichlet_mean": "dirichlet_mean"})
    # df2['energy_time'] = 'TN'
    # df_cat = pd.concat([df1, df2])
    # fig, ax = plt.subplots()
    # ax.set_title(f"T0->TN_dirichlet_energy_mean", fontdict={'fontsize': 24})
    # sns.stripplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", jitter=False, data=df_cat, ax=ax)
    # # sns.scatterplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", data=df_cat, ax=ax)
    # fig.show()

    # #regulaised energy
    # fig, ax = plt.subplots()
    # ax.set_title(f"T0r_dirichlet_energy_mean", fontdict={'fontsize': 24})
    # sns.stripplot(x="target_homoph", y="T0r_dirichlet_mean", hue="gnl_W_style", jitter=False, data=max_df, ax=ax)
    # fig.show()
    # fig, ax = plt.subplots()
    # ax.set_title(f"TNr_dirichlet_energy_mean", fontdict={'fontsize': 24})
    # sns.stripplot(x="target_homoph", y="TNr_dirichlet_mean", hue="gnl_W_style", jitter=0.15, data=max_df, ax=ax)
    # fig.show()

    new_cols = ["target_homoph", "gnl_W_style", "T0r_dirichlet_mean"]
    df1 = max_df[new_cols]
    df1 = df1.rename(columns={"T0r_dirichlet_mean": "dirichlet_mean"})
    df1['energy_time'] = 'T0'
    new_cols = ["target_homoph", "gnl_W_style", "TNr_dirichlet_mean"]
    df2 = max_df[new_cols]
    df2 = df2.rename(columns={"TNr_dirichlet_mean": "dirichlet_mean"})
    df2['energy_time'] = 'TN'

    if ax is None:
        fig, ax = plt.subplots()
        ax_idx = 0
    ax.set_title(f"T0->TN dirichlet_energy_reg_mean", fontdict={'fontsize': 24})
    df_cat = pd.concat([df1, df2])
    sns.scatterplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", style="energy_time", data=df_cat, ax=ax[ax_idx])
    # sns.stripplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", marker="o", data=df1, jitter=0.15, ax=ax)
    # sns.stripplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", marker="X", data=df2, jitter=0.15, ax=ax)
    x_bins = None
    y_bins = None
    estimator = None
    ci = 95
    n_boot = 1000,
    alpha = None
    x_jitter = None
    y_jitter = None,
    legend = "auto"
    variables = _ScatterPlotter.get_semantics(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", style="energy_time", data=df_cat, ax=ax)
    sp = _ScatterPlotter(
        data=data, variables=variables,
        x_bins=x_bins, y_bins=y_bins,
        estimator=estimator, ci=ci, n_boot=n_boot,
        alpha=alpha, x_jitter=x_jitter, y_jitter=y_jitter, legend=legend)

    # SP = _ScatterPlotter(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", style="energy_time", data=df_cat, ax=ax)
    sp.add_legend_data(ax)
    fig.show()

def syn_cora_homoph(path, line_scatter, fig=None, ax=None, ax_idx=None, plot=False, save=False):
    max_df = get_max_df(path)
    max_df = max_df.replace(to_replace={"diag_dom":"diag-dom", "neg_prod":"neg-prod"})
    max_df = max_df.rename(columns={"gnl_W_style": "W-style"})    # maxdf ####need tp rename column
    new_cols = ["target_homoph", "W-style", "enc_pred_homophil"]#, "label_homophil_mean", "pred_homophil_mean"]
    df1 = max_df[new_cols]
    df1 = df1.rename(columns={"enc_pred_homophil": "homophily"})
    df1['module-block'] = 'encoder'
    new_cols = ["target_homoph", "W-style", "pred_homophil_mean"]
    df2 = max_df[new_cols]
    df2 = df2.rename(columns={"pred_homophil_mean": "homophily"})
    df2['module-block'] = 'prediction'
    df = pd.concat([df1, df2])
    df = df.reset_index(drop=True)

    if line_scatter == "scatter":
        # manually jitter
        mask = (df["W-style"] == "sum")
        df.loc[mask, 'target_homoph'] = df.loc[mask, 'target_homoph'] + 0.005
        mask = (df["W-style"] == "diag-dom")
        df.loc[mask, 'target_homoph'] = df.loc[mask, 'target_homoph'] - 0.005

    fs = 16
    ps = 150
    if ax is None:
        fig, ax = plt.subplots()
        if line_scatter == "scatter":
            sns.scatterplot(x="target_homoph", y="homophily", hue="W-style", style="module-block", s=ps, data=df,
                            ax=ax)
        elif line_scatter == "line":
            sns.lineplot(x="target_homoph", y="homophily", hue="W-style", style="module-block", data=df, ax=ax)
        ax.set_xlabel('Target homophily', fontsize=fs)
        ax.set_ylabel('Homophily', fontsize=fs)
        ax.legend(prop=dict(size=fs-4), loc='upper left')
    else:
        if line_scatter == "scatter":
            sns.scatterplot(x="target_homoph", y="homophily", hue="W-style", style="module-block", s=ps, data=df,
                            ax=ax[ax_idx])
        elif line_scatter == "line":
            sns.lineplot(x="target_homoph", y="homophily", hue="W-style", style="module-block", data=df, ax=ax[ax_idx])

        ax[ax_idx].set_xlabel('Target homophily', fontsize=fs)
        ax[ax_idx].set_ylabel('Homophily', fontsize=fs)
        ax[ax_idx].legend(prop=dict(size=fs-4), loc='upper left')

    # ax[ax_idx].set_title(f"T0->TN homophily", fontdict={'fontsize': 18})
    # sns.stripplot(x="target_homoph", y="homophily", hue="W-style", marker="o", data=df1, jitter=0.15, ax=ax)
    # sns.stripplot(x="target_homoph", y="homophily", hue="W-style", marker="X", data=df2, jitter=0.15, ax=ax)
    if save:
        plt.savefig('../ablations/syn_cora_homoph.pdf')
    if plot:
        fig.show()
    return fig, ax

def plot_1(path, line_scatter, plot=True, save=True):
    sns.set_theme()
    fig, ax = plt.subplots(2,1,figsize=(10, 10), sharex=True)
    fig, ax = syn_cora_plot(path, fig, ax, ax_idx=0, plot=False, save=False)
    fig, ax = syn_cora_homoph(path, line_scatter, fig, ax, ax_idx=1, plot=False, save=False)

    # ax[0].get_shared_x_axes().join(ax[0], ax[1])
    # ax[0].set_xticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0.015)
    fig.tight_layout()
    if save:
        plt.savefig(f"../ablations/plot_1_{line_scatter}.pdf", bbox_inches='tight')
    if plot:
        fig.show()

def wall_clock(path, model, line_scatter="scatter", plot=True, save=True):
    sns.set_theme()
    sns.color_palette()
    df = pd.read_csv(path)
    if model == "greed_non_linear":
        df = df[(df.function == "greed_non_linear")].reset_index(drop=True)
    else:
        df = df[(df.function != "greed_non_linear")].reset_index(drop=True)
        replace_dict = {0:"0:gcn",1:"1:gcn_enc/dec",2:"2:gcn_residual",3:"3:gcn_share_W",4:"4:gcn_symm_W",5:"GraF"}#"5:no_nonLin"}
        df.loc[:,'gcn_params_idx'].replace(to_replace=replace_dict, inplace=True)

        mask = (df["function"] == "gat")
        df.loc[mask, 'gcn_params_idx'] = "gat"
        mask = (df["function"] == "gcn")
        df.loc[mask, 'gcn_params_idx'] = "pyg-gcn"

    fs = 14
    ps = 25
    fig, ax = plt.subplots()
    if model == "greed_non_linear":
        if line_scatter == "scatter":
            sns.scatterplot(x="hidden_dim", y="av_fwd", hue="gnl_W_style", s=ps, data=df,
                            ax=ax, palette="deep")#, marker="x")
        elif line_scatter == "line":
            sns.lineplot(x="hidden_dim", y="av_fwd", hue="gnl_W_style", data=df, ax=ax)
    else:
        if line_scatter == "scatter":
            sns.scatterplot(x="hidden_dim", y="av_fwd", hue="gcn_params_idx", s=ps, data=df,
                            ax=ax, palette="deep")#, marker="x")
        elif line_scatter == "line":
            sns.lineplot(x="hidden_dim", y="av_fwd", hue="gcn_params_idx", data=df, ax=ax)
    ax.set_xlabel('hidden dim', fontsize=fs)
    ax.set_ylabel('runtime', fontsize=fs)
    ax.legend(prop=dict(size=fs-4), loc='upper left')
    if save:
        plt.savefig('../ablations/wall_clock_runtime.pdf')
    if plot:
        fig.show()

    fig, ax = plt.subplots()
    if line_scatter == "scatter":
        sns.scatterplot(x="hidden_dim", y="num_params", hue="gcn_params_idx", s=ps, data=df,
                        ax=ax, palette="deep")#, marker="x")
    elif line_scatter == "line":
        sns.lineplot(x="hidden_dim", y="num_params", hue="gcn_params_idx", data=df, ax=ax)
    ax.set_xlabel('Hidden dim', fontsize=fs)
    ax.set_ylabel('# params', fontsize=fs)
    ax.legend(prop=dict(size=fs-4), loc='upper left')
    if save:
        plt.savefig('../ablations/wall_clock_params.pdf')
    if plot:
        fig.show()


if __name__ == "__main__":
    # path = "../ablations/ablation_syn_cora.csv"
    # _,_ = syn_cora_plot(path, plot=True, save=True)
    # syn_cora_best_times(path)
    # syn_cora_energy(path)
    # _,_ = syn_cora_homoph(path)
    # plot_1(path, "scatter")
    # plot_1(path, "line")
    # _,_ = syn_cora_gcn_plot(path="../ablations/ablation_syn_cora_gcn.csv", plot=True, save=True)
    # wall_clock(path="../ablations/wallclock.csv", model="gcn")
    # wall_clock(path="../ablations/wallclock.csv", model="greed_non_linear")
    size_d_plot(path="../ablations/ablation_size_d2.csv")