from optimization_algorithms import run_optimization, gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent
from visualize_results import plot_results

# Get user input for initial_value and num_iterations
initial_value = float(input("Enter the initial value for optimization: "))
num_iterations = int(input("Enter the number of iterations: "))

# Run optimization using different algorithms
gd_results = run_optimization(gradient_descent, initial_value, num_iterations)
sgd_results = run_optimization(stochastic_gradient_descent, initial_value, num_iterations)
mbgd_results = run_optimization(mini_batch_gradient_descent, initial_value, num_iterations)

# Plot the results
plot_results([gd_results, sgd_results, mbgd_results], ['Gradient Descent', 'Stochastic Gradient Descent', 'Mini-Batch Gradient Descent'])

# ... Add more plots for additional algorithms ...
