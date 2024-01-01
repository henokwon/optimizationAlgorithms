from flask import Flask, render_template, request
import os
import numpy as np
from optimization_algorithms import run_optimization, gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent
from visualize_results import plot_results

app = Flask(__name__)

def optimize_and_plot(initial_value, num_iterations, algorithm):
    algorithms = {
        'gd': gradient_descent,
        'sgd': stochastic_gradient_descent,
        'mbgd': mini_batch_gradient_descent,
        # Add more entries for additional algorithms
    }

    selected_algorithm = algorithms.get(algorithm)
    if selected_algorithm:
        results = run_optimization(selected_algorithm, initial_value, num_iterations)
        plot_results([results], [algorithm.capitalize()], save_as='static/optimization_plot.png')

@app.route('/', methods=['GET', 'POST'])
def home():
    algorithm_mappings = {
        'gd': 'Gradient Descent',
        'sgd': 'Stochastic Gradient Descent',
        'mbgd': 'Mini-Batch Gradient Descent',
        # Add more entries for additional algorithms
    }
    
    algorithms = list(algorithm_mappings.keys())  # Use abbreviations as keys
    if request.method == 'POST':
        initial_value = float(request.form['initial_value'])
        num_iterations = int(request.form['num_iterations'])
        
        # Ensure 'algorithm' is present in the form data before accessing it
        selected_algorithm = request.form.get('algorithm')

        if selected_algorithm and selected_algorithm in algorithms:
            optimize_and_plot(initial_value, num_iterations, selected_algorithm)
            return render_template('index.html', image_path='static/optimization_plot.png', show_image=True, algorithms=algorithm_mappings, selected_algorithm=selected_algorithm)
        else:
            return render_template('index.html', image_path=None, show_image=False, algorithms=algorithm_mappings, selected_algorithm=None, error_message="Invalid algorithm selected.")
    
    return render_template('index.html', image_path=None, show_image=False, algorithms=algorithm_mappings, selected_algorithm=None)

if __name__ == '__main__':
    app.run(debug=True)
