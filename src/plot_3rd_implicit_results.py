import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import cm

# Settings
dataset = 'Cora'
method = 'implicit_adams'
count_runs = 10
topK = 4
#stepsizes = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
results = {}
combined_results = {}

# Load implicit_adams saved results
stepsize = 1.0
for topK in [4, 8, 16, 32, 64]:
    results[topK] = []
    for idx in range(0, count_runs):
        results[topK].append( pickle.load( open( f"../results/{dataset}_{method}_stepsize_{stepsize}_rewiring_topK_{topK}_run_{idx}.pickle", "rb" ) ))

# Combine runs from implicit_adams results
for topK, result in results.items():
    combined_results[topK] = {}
    for k in result[0].keys():
        values = np.array([result[idx][k] for idx in range(len(result))])
        combined_results[topK][k] = values

# Load explicit_adams saved results
results = {}
combined_explicit_results = {}
method = 'explicit_adams'
stepsize = 0.005
for topK in [4, 8, 16, 32, 64]:
    results[topK] = []
    for idx in range(0, count_runs):
        results[topK].append( pickle.load( open( f"../results/{dataset}_{method}_stepsize_{stepsize}_rewiring_topK_{topK}_run_{idx}.pickle", "rb" ) ))

# Combine runs from explicit_adams results
for topK, result in results.items():
    combined_explicit_results[topK] = {}
    for k in result[0].keys():
        values = np.array([result[idx][k] for idx in range(len(result))])
        combined_explicit_results[topK][k] = values

# Load dopri5
results = {}
combined_dopri_results = {}
for topK in [4, 8, 16, 32, 64]:
    results[topK] = []
    for idx in range(0, count_runs):
        results[topK].append( pickle.load( open( f"../results/{dataset}_dopri5_rewiring_topK_{topK}_run_{idx}.pickle", "rb" ) ))

# Combine runs from results
for topK, result in results.items():
    combined_dopri_results[topK] = {}
    for k in result[0].keys():
        values = np.array([result[idx][k] for idx in range(len(result))])
        combined_dopri_results[topK][k] = values

# Plot Test Accuracy
plt.figure(figsize=(8,5), dpi= 100)
#plt.title(f'Cora - Test Accuracy - K = {topK}')

count_K = 4
for idx, topK in enumerate([4, 8, 16, 32]):
    count_runs = combined_results[topK]['test_acc'].shape[0]
    #count_epochs = combined_results[topK]['test_acc'].shape[1]

    # Plot explicit
    x = np.cumsum(np.mean(combined_results[topK]['time'], axis=0))
    y = np.mean(combined_results[topK]['test_acc'], axis=0)
    std = np.std(combined_results[topK]['test_acc'], axis=0)
    plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_K), alpha=0.25)
    plt.plot(x, y, label=f'imp-adams $K={topK}$', color=cm.jet(idx/count_K))

    # Plot implicit
    x = np.cumsum(np.mean(combined_explicit_results[topK]['time'], axis=0))
    y = np.mean(combined_explicit_results[topK]['test_acc'], axis=0)
    std = np.std(combined_explicit_results[topK]['test_acc'], axis=0)
    plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_K), alpha=0.25)
    plt.plot(x, y, label=f'exp-adams $K={topK}$', linestyle=":", color=cm.jet(idx/count_K))

    # Plot dopri5
    x = np.cumsum(np.mean(combined_dopri_results[topK]['time'], axis=0))
    y = np.mean(combined_dopri_results[topK]['test_acc'], axis=0)
    std = np.std(combined_dopri_results[topK]['test_acc'], axis=0)
    plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.gray(idx/count_K), alpha=0.25)
    plt.plot(x, y, label=f'dopri5 $K={topK}$', linestyle=":", color=cm.gray(idx/count_K))

#plt.xlim((-3.0,7.0))
#plt.ylim((0.1,0.84))
plt.xlabel('clock time (sec)')
plt.ylabel('acc (ratio)')
plt.xscale('log')
plt.legend()
plt.savefig(f'cora-rewiring-test-acc-vs-clock-time.pdf', bbox_inches='tight')
