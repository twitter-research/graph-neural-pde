import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


df_rewiring_node = pd.read_csv('../results/GPU_rewiring_node_100seeds.csv',header=[0,1,2])
pd.options.display.max_rows = 50 #None
df_rewiring_node

ks = [1 ,2 ,4 ,8 ,16 ,32 ,64 ,128 ,256]
fig =plt.figure(figsize=(16 ,16))
for rc, i in enumerate(ks):
    print(f"target k {i}")
    k = df_rewiring_node['Unnamed: 2_level_0']['Unnamed: 2_level_1']['k']
    idx = k.index[k == i].tolist()
    print(idx)
    orig = df_rewiring_node['Unnamed: 3_level_0']['Unnamed: 3_level_1']['orig_edges'].iloc[idx]
    count = df_rewiring_node['final_edges']['count']['Unnamed: 5_level_2'].iloc[idx]
    # means
    added = df_rewiring_node['added']['mean']['Unnamed: 4_level_2'].iloc[idx]
    retained = df_rewiring_node['orig_retained']['mean']['Unnamed: 8_level_2'].iloc[idx]
    removed = df_rewiring_node['orig_removed']['mean']['Unnamed: 7_level_2'].iloc[idx]

    # get indices matching buckets
    bin_list = []
    bin_labels = []
    for j, l in enumerate(ks[:len(ks ) -1]):
        #         print(j,l,ks[j+1])
        idx_list = orig.index[(orig >= l) & (orig < ks[ j +1])].tolist()
        bin_list.append(idx_list)
        bin_labels.append(f"{l}:<{ks[ j +1]}")
    if orig.index[orig >= ks[-1]].tolist() != []:
        bin_list.append(orig.index[orig >= ks[-1]].tolist())
        bin_labels.append(f">{ks[-1]}")

    print(bin_list)
    print(bin_labels)

    # find weighted mean of the 3 parts for each bucket
    wav_added = []
    wav_retained = []
    wav_removed = []
    for idxs in bin_list:
        print(idxs)
        print(f"count[idxs]::: {count[idxs]}")
        print(f"add {(added[idxs] * count[idxs]).sum( ) /count[idxs].sum()}")
        print(f"ret {(retained[idxs] * count[idxs]).sum( ) /count[idxs].sum()}")
        print(f"rem {(removed[idxs] * count[idxs]).sum( ) /count[idxs].sum()}")
        # wav_added.append((added[idxs] * count[idxs]).sum( ) /count[idxs].sum())
        # wav_retained.append((retained[idxs] * count[idxs]).sum( ) /count[idxs].sum())
        # wav_removed.append((removed[idxs] * count[idxs]).sum( ) /count[idxs].sum())

#     bar_df = pd.DataFrame(index=bin_labels,
#                           data={'retained': wav_retained,
#                                 'added': wav_added,
#                                 'removed': wav_removed})
#     ax=fig.add_subplot(3 ,3 ,rc+1)
#     bar_df.plot(kind="bar", stacked=True, ax=ax)
#     ax.set_title(f"target degree {i}")
# plt.show()