import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib. pyplot as plt

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

def syn_cora_plot(path):
    df = pd.read_csv(path)
    gnl_df = df[(df.function == "greed_non_linear")]
    gnl_piv = pd.pivot_table(gnl_df, values="test_mean", index="target_homoph", columns="gnl_W_style", aggfunc=np.max)
    base_df = df[(df.function != "greed_non_linear")]
    base_piv = pd.pivot_table(base_df, values="test_mean", index="target_homoph", columns="function", aggfunc=np.max)
    piv = pd.merge(gnl_piv, base_piv, on=['target_homoph'])
    fig, ax = plt.subplots()
    sns.lineplot(data=piv, palette="tab10", linewidth=2.5)
    fig.show()

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

def syn_cora_energy(path):
    max_df = get_max_df(path)
    # max_df['W_homoph'] = max_df.apply(lambda row: str(row.gnl_W_style) + "_" + str(row.target_homoph), axis=1)

    fig, ax = plt.subplots()
    ax.set_title(f"T0_dirichlet_energy_mean", fontdict={'fontsize': 24})
    sns.stripplot(x="target_homoph", y="T0_dirichlet_mean", hue="gnl_W_style", jitter=False, data=max_df, ax=ax)
    fig.show()
    fig, ax = plt.subplots()
    ax.set_title(f"TN_dirichlet_energy_mean", fontdict={'fontsize': 24})
    sns.stripplot(x="target_homoph", y="TN_dirichlet_mean", hue="gnl_W_style", jitter=0.15, data=max_df, ax=ax)
    fig.show()

    new_cols = ["target_homoph", "gnl_W_style", "T0_dirichlet_mean"]
    df1 = max_df[new_cols]
    df1 = df1.rename(columns={"T0_dirichlet_mean": "dirichlet_mean"})
    df1['energy_time'] = 'T0'
    new_cols = ["target_homoph", "gnl_W_style", "TN_dirichlet_mean"]
    df2 = max_df[new_cols]
    df2 = df2.rename(columns={"TN_dirichlet_mean": "dirichlet_mean"})
    df2['energy_time'] = 'TN'
    df_cat = pd.concat([df1, df2])
    fig, ax = plt.subplots()
    ax.set_title(f"T0->TN_dirichlet_energy_mean", fontdict={'fontsize': 24})
    sns.stripplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", jitter=False, data=df_cat, ax=ax)
    # sns.scatterplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", data=df_cat, ax=ax)
    fig.show()


    fig, ax = plt.subplots()
    ax.set_title(f"T0r_dirichlet_energy_mean", fontdict={'fontsize': 24})
    sns.stripplot(x="target_homoph", y="T0r_dirichlet_mean", hue="gnl_W_style", jitter=False, data=max_df, ax=ax)
    fig.show()
    fig, ax = plt.subplots()
    ax.set_title(f"TNr_dirichlet_energy_mean", fontdict={'fontsize': 24})
    sns.stripplot(x="target_homoph", y="TNr_dirichlet_mean", hue="gnl_W_style", jitter=0.15, data=max_df, ax=ax)
    fig.show()

    new_cols = ["target_homoph", "gnl_W_style", "T0r_dirichlet_mean"]
    df1 = max_df[new_cols]
    df1 = df1.rename(columns={"T0r_dirichlet_mean": "dirichlet_mean"})
    df1['energy_time'] = 'T0'
    new_cols = ["target_homoph", "gnl_W_style", "TNr_dirichlet_mean"]
    df2 = max_df[new_cols]
    df2 = df2.rename(columns={"TNr_dirichlet_mean": "dirichlet_mean"})
    df2['energy_time'] = 'TN'
    df_cat = pd.concat([df1, df2])
    fig, ax = plt.subplots()
    ax.set_title(f"T0->TN_dirichlet_energy_reg_mean", fontdict={'fontsize': 24})
    sns.scatterplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", style="energy_time", data=df_cat, ax=ax)
    # sns.stripplot(x="target_homoph", y="dirichlet_mean", hue="gnl_W_style", marker="energy_time", data=df_cat, jitter=0.15, ax=ax)
    fig.show()
    print("yo")

if __name__ == "__main__":
    path = "../ablations/ablation_syn_cora.csv"
    # syn_cora_plot(path)
    # syn_cora_best_times(path)
    syn_cora_energy(path)