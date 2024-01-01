import numpy as np

# Define the quadratic function
def quadratic_function(x):
    return x**2 + 4*x + 4

# Define the gradient of the quadratic function
def gradient_quadratic_function(x):
    return 2*x + 4

# Implement different optimization algorithms
def gradient_descent(x, learning_rate=0.1):
    return x - learning_rate * gradient_quadratic_function(x)

def stochastic_gradient_descent(x, learning_rate=0.1):
    noise = np.random.normal(0, 1)
    return x - learning_rate * gradient_quadratic_function(x + noise)

def mini_batch_gradient_descent(x, batch_size=10, learning_rate=0.1):
    noise = np.random.normal(0, 1, batch_size)
    return x - learning_rate * np.mean(gradient_quadratic_function(x + noise))

# ... Add more optimization algorithms as needed ...

# Run optimization for a specified number of iterations
def run_optimization(optimizer, initial_value, num_iterations):
    values = []
    x = initial_value
    for _ in range(num_iterations):
        x = optimizer(x)
        values.append(quadratic_function(x))
    return values
