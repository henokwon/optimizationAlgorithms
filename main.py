from optimization_algorithms import run_optimization
from visualize_results import plot_results

# Set up the optimization parameters
initial_value = -8
num_iterations = 100

# Run optimization using different algorithms
gd_results = run_optimization(gradient_descent, initial_value, num_iterations)
sgd_results = run_optimization(stochastic_gradient_descent, initial_value, num_iterations)
mbgd_results = run_optimization(mini_batch_gradient_descent, initial_value, num_iterations)

# ... Run more optimization algorithms as needed ...

# Plot the results
plot_results([gd_results, sgd_results, mbgd_results], ['Gradient Descent', 'Stochastic Gradient Descent', 'Mini-Batch Gradient Descent'])

# ... Add more plots for additional algorithms ...
