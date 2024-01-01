import matplotlib.pyplot as plt

# Visualize the results
def plot_results(opt_results, labels):
    for results, label in zip(opt_results, labels):
        plt.plot(results, label=label)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value')
    plt.legend()
    plt.show()
