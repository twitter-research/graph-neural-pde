import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import cm

# Settings
dataset = 'Cora'
method = 'implicit_adams'
count_runs = 10
topK = 16
stepsizes = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
results = {}
combined_results = {}

# Load implicit_adams saved results
for stepsize in stepsizes:
    results[stepsize] = []
    for idx in range(0, count_runs):
        results[stepsize].append( pickle.load( open( f"../results/{dataset}_{method}_stepsize_{stepsize}_rewiring_topK_{topK}_run_{idx}.pickle", "rb" ) ))

# Combine runs from implicit_adams results
for stepsize, result in results.items():
    combined_results[stepsize] = {}
    for k in result[0].keys():
        values = np.array([result[idx][k] for idx in range(len(result))])
        combined_results[stepsize][k] = values

# Load explicit_adams saved results
results = {}
combined_explicit_results = {}
method = 'explicit_adams'
explicit_stepsizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.005]
for stepsize in explicit_stepsizes:
    results[stepsize] = []
    for idx in range(0, count_runs):
        results[stepsize].append( pickle.load( open( f"../results/{dataset}_{method}_stepsize_{stepsize}_rewiring_topK_{topK}_run_{idx}.pickle", "rb" ) ))

# Combine runs from explicit_adams results
for stepsize, result in results.items():
    combined_explicit_results[stepsize] = {}
    for k in result[0].keys():
        values = np.array([result[idx][k] for idx in range(len(result))])
        combined_explicit_results[stepsize][k] = values

# Load
results_dopri = []
for idx in range(0, count_runs):
    results_dopri.append( pickle.load( open( f"../results/{dataset}_dopri5_rewiring_topK_{topK}_run_{idx}.pickle", "rb" ) ))

# Combine runs from results
combined_results_dopri = {}
for k in results_dopri[0].keys():
    values = np.array([results_dopri[idx][k] for idx in range(len(results_dopri))])
    combined_results_dopri[k] = values

# Plot Test Accuracy
plt.figure(figsize=(8,5), dpi= 100)
#plt.title(f'Cora - Test Accuracy - K = {topK}')
    
count_stepsizes = len([5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.005])
for idx, stepsize in enumerate([5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.005]):
    if stepsize in combined_results:
        count_runs = combined_results[stepsize]['test_acc'].shape[0]
        count_epochs = combined_results[stepsize]['test_acc'].shape[1]

        x = np.cumsum(np.mean(combined_results[stepsize]['time'], axis=0))
        y = np.mean(combined_results[stepsize]['test_acc'], axis=0)
        std = np.std(combined_results[stepsize]['test_acc'], axis=0)

        # Plot +- 1 standard error
        plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_stepsizes), alpha=0.25)
        plt.plot(x, y, label=f'imp $dt={stepsize}$', color=cm.jet(idx/count_stepsizes))
    
    if stepsize in combined_explicit_results:
        count_runs = combined_explicit_results[stepsize]['test_acc'].shape[0]
        count_epochs = combined_explicit_results[stepsize]['test_acc'].shape[1]

        x = np.cumsum(np.mean(combined_explicit_results[stepsize]['time'], axis=0))
        y = np.mean(combined_explicit_results[stepsize]['test_acc'], axis=0)
        std = np.std(combined_explicit_results[stepsize]['test_acc'], axis=0)

        # Plot +- 1 standard error
        plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_stepsizes), alpha=0.25)
        plt.plot(x, y, label=f'exp $dt={stepsize}$', linestyle=':', color=cm.jet(idx/count_stepsizes))

x = np.cumsum(np.mean(combined_results_dopri['time'], axis=0))
y = np.mean(combined_results_dopri['test_acc'], axis=0)
std = np.std(combined_results_dopri['test_acc'], axis=0)
plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color='black', alpha=0.25)
plt.plot(x, y, label=f'dopri5', color='black')
#plt.xlim((-3.0,7.0))
#plt.ylim((0.1,0.84))
plt.xlabel('clock time (sec)')
plt.ylabel('acc (ratio)')
plt.xscale('log')
plt.legend()
plt.savefig(f'cora-no-rewiring-test-acc-vs-clock-time.pdf', bbox_inches='tight')
