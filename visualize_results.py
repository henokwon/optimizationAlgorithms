# visualize_results.py
import matplotlib.pyplot as plt
import os

# Visualize the results
def plot_results(opt_results, labels, save_as=None):
    for results, label in zip(opt_results, labels):
        plt.plot(results, label=label)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value')
    plt.legend()

    # Save the figure if save_as is provided
    if save_as:
        plt.savefig(save_as)
        plt.close()
    else:
        plt.show()
