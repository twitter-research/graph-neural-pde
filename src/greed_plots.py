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

# Import Data
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")
# Draw Stripplot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
sns.stripplot(df.cty, df.hwy, jitter=0.25, size=8, ax=ax, linewidth=.5)
# Decorations
plt.title('Use jittered plots to avoid overlapping of points', fontsize=22)
# plt.show()


df = pd.read_csv("~/Downloads/wandb_export_2022-05-07T14_06_35.099+01_00.csv")
piv = pd.pivot_table(df, values="test_mean", index="target_homoph", columns="gnl_W_style", aggfunc=np.max)
fig, ax = plt.subplots()
sns.lineplot(data=piv, palette="tab10", linewidth=2.5)
fig.show()

print(df)
for target_homoph in ['0.00', '0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00']:
    for w_type in ["diag_dom", "neg_prod", "prod", "sum"]:
        new_cols = ["target_homoph", "gnl_W_style", "time", "step_size", "test_mean"]
        newdf = df[new_cols][(df.target_homoph == float(target_homoph)) & (df.gnl_W_style == w_type)]
        max_idx = newdf[['test_mean']].idxmax()
        max_time = df.iloc[max_idx]['time'] #.values
        max_step = df.iloc[max_idx]['step_size'] #.values

