from flask import Flask, render_template, request
import numpy as np
from optimization_algorithms import run_optimization, gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent
from visualize_results import plot_results

app = Flask(__name__)

def optimize_and_plot(initial_value, num_iterations):
    gd_results = run_optimization(gradient_descent, initial_value, num_iterations)
    sgd_results = run_optimization(stochastic_gradient_descent, initial_value, num_iterations)
    mbgd_results = run_optimization(mini_batch_gradient_descent, initial_value, num_iterations)

    # Save the plot as an image file
    plot_results([gd_results, sgd_results, mbgd_results], ['Gradient Descent', 'Stochastic Gradient Descent', 'Mini-Batch Gradient Descent'], save_as='static/optimization_plot.png')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        initial_value = float(request.form['initial_value'])
        num_iterations = int(request.form['num_iterations'])
        optimize_and_plot(initial_value, num_iterations)
        return render_template('index.html', image_path='static/optimization_plot.png')
    return render_template('index.html', image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
