import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import cm

# Settings
dataset = 'Cora'
method = 'implicit_adams'
count_runs = 10
stepsizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.01]
results = {}
combined_results = {}

# Load saved results
for stepsize in stepsizes:
    results[stepsize] = []
    for idx in range(0, count_runs):
        results[stepsize].append( pickle.load( open( f"../results/{dataset}_{method}_stepsize_{stepsize}_run_{idx}.pickle", "rb" ) ))

# Combine runs from results
for stepsize, result in results.items():
    combined_results[stepsize] = {}
    for k in result[0].keys():
        values = np.array([result[idx][k] for idx in range(len(result))])
        combined_results[stepsize][k] = values

# DEBUG
#print(combined_results[0.5]['loss'].shape)

# Plot Losses
plt.figure(figsize=(9,6), dpi= 100)
plt.title('Cora - Train Loss')

count_stepsizes = len(stepsizes)
for idx, stepsize in enumerate(stepsizes):
    count_runs = combined_results[stepsize]['loss'].shape[0]
    count_epochs = combined_results[stepsize]['loss'].shape[1]

    x = np.arange(0, count_epochs)
    y = np.mean(combined_results[stepsize]['loss'], axis=0)
    std = np.std(combined_results[stepsize]['loss'], axis=0)

    # Plot +- 1 standard error
    plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_stepsizes), alpha=0.25)
    plt.plot(y, label=f'$dx={stepsize}$', color=cm.jet(idx/count_stepsizes))

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('cora-implicit-adams-loss.png')

# Plot Times
plt.figure(figsize=(9,6), dpi= 100)
plt.title('Cora - Time per Epoch')

count_stepsizes = len(stepsizes)
for idx, stepsize in enumerate(stepsizes):
    count_runs = combined_results[stepsize]['time'].shape[0]
    count_epochs = combined_results[stepsize]['time'].shape[1]

    x = np.arange(0, count_epochs)
    y = np.mean(combined_results[stepsize]['time'], axis=0)
    std = np.std(combined_results[stepsize]['time'], axis=0)

    # Plot +- 1 standard error
    plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_stepsizes), alpha=0.25)
    plt.plot(y, label=f'$dx={stepsize}$', color=cm.jet(idx/count_stepsizes))

plt.xlabel('epochs')
plt.ylabel('time (sec)')
plt.legend()
plt.savefig('cora-implicit-adams-time.png')

# Plot Times (zoomed)
plt.figure(figsize=(9,6), dpi= 100)
plt.title('Cora - Time per Epoch (Zoomed)')

count_stepsizes = len(stepsizes)
for idx, stepsize in enumerate(stepsizes):
    count_runs = combined_results[stepsize]['time'].shape[0]
    count_epochs = combined_results[stepsize]['time'].shape[1]

    x = np.arange(0, count_epochs)
    y = np.mean(combined_results[stepsize]['time'], axis=0)
    std = np.std(combined_results[stepsize]['time'], axis=0)

    # Plot +- 1 standard error
    plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_stepsizes), alpha=0.25)
    plt.plot(y, label=f'$dx={stepsize}$', color=cm.jet(idx/count_stepsizes))

plt.xlabel('epochs')
plt.ylabel('time (sec)')
plt.ylim((0, 2.0))
plt.legend()
plt.savefig('cora-implicit-adams-time-zoomed.png')

# Plot Test Accuracy
plt.figure(figsize=(9,6), dpi= 100)
plt.title('Cora - Test Accuracy')

count_stepsizes = len(stepsizes)
for idx, stepsize in enumerate(stepsizes):
    count_runs = combined_results[stepsize]['test_acc'].shape[0]
    count_epochs = combined_results[stepsize]['test_acc'].shape[1]

    x = np.arange(0, count_epochs)
    y = np.mean(combined_results[stepsize]['test_acc'], axis=0)
    std = np.std(combined_results[stepsize]['test_acc'], axis=0)

    # Plot +- 1 standard error
    plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_stepsizes), alpha=0.25)
    plt.plot(y, label=f'$dx={stepsize}$', color=cm.jet(idx/count_stepsizes))

plt.xlabel('epochs')
plt.ylabel('acc (ratio)')
plt.legend()
plt.savefig('cora-implicit-adams-test-acc.png')

# Plot Test Accuracy (zoomed)
plt.figure(figsize=(9,6), dpi= 100)
plt.title('Cora - Test Accuracy (Zoomed)')

count_stepsizes = len(stepsizes)
for idx, stepsize in enumerate(stepsizes):
    count_runs = combined_results[stepsize]['test_acc'].shape[0]
    count_epochs = combined_results[stepsize]['test_acc'].shape[1]

    x = np.arange(0, count_epochs)
    y = np.mean(combined_results[stepsize]['test_acc'], axis=0)
    std = np.std(combined_results[stepsize]['test_acc'], axis=0)

    # Plot +- 1 standard error
    plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color=cm.jet(idx/count_stepsizes), alpha=0.25)
    plt.plot(y, label=f'$dx={stepsize}$', color=cm.jet(idx/count_stepsizes))

plt.xlabel('epochs')
plt.ylabel('acc (ratio)')
plt.ylim(0.75,0.85)
plt.legend()
plt.savefig('cora-implicit-adams-test-acc-zoomed.png')

# Plot Best Test Accuracy vs Stepsize
plt.figure(figsize=(9,6), dpi= 100)
plt.title('Cora - Test Accuracy vs Stepsize')

count_stepsizes = len(stepsizes)
best_acc = []
for idx, stepsize in enumerate(stepsizes):
    count_runs = combined_results[stepsize]['test_acc'].shape[0]
    #count_epochs = combined_results[stepsize]['test_acc'].shape[1]
    #x = np.arange(0, count_epochs)

    best_acc.append(combined_results[stepsize]['best_test_acc'])
    
best_acc = np.array(best_acc)
#print(best_acc.shape)

x = np.array(stepsizes)
y = np.mean(best_acc, axis=1)
std = np.std(best_acc, axis=1)

# Plot +- 1 standard error
plt.fill_between(x, y - std/math.sqrt(count_runs), y + std/math.sqrt(count_runs), color='blue', alpha=0.25)
plt.plot(x, y, label=f'$dx={stepsize}$', color='blue', marker='.')

plt.xlabel('stepsize')
plt.ylabel('acc (ratio)')
#plt.ylim(0.75,0.85)
plt.legend()
plt.savefig('cora-implicit-adams-test-acc-vs-stepsize.png')