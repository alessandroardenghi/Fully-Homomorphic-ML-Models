import numpy as np
import matplotlib.pyplot as plt

# Function to define the objective function as difference between y and a given polynomial
def objective_function(coefficients, x, y, weights):
    polynomial = np.poly1d(coefficients)
    approximation = polynomial(x)
    error = approximation - y
    squared_error = np.square(error)
    weighted_error = np.multiply(squared_error, weights)
    return np.sum(weighted_error)

# Definition of the step function
def step_function(x):
    return np.where(x < 0, 0, 1)

# Definition of the weight function
def weight(window, x):

  weight_in_window = 0.0
  weight_outside_window = 1.0
  weights = np.where(np.abs(x) <= window, weight_in_window, weight_outside_window)
  integral_weights = np.trapz(weights, x)
  weights /= integral_weights
  return weights

# Function to obtain a polynomial approxiamtion of "function" through L2 optimisation
def poly_approx(function, degree, interval, window):
    num_points = 1000
    x = np.linspace(-interval, interval, num_points)
    y = function(x)
    w = weight(window, x)

    X = np.vander(x, degree + 1)  # Vandermonde matrix of x
    W = np.diag(np.sqrt(w))  # Weight matrix

    # Perform L2 optimization
    coefficients, _, _, _ = np.linalg.lstsq(W @ X, np.sqrt(w) * y, rcond=None)
    polynomial = np.poly1d(coefficients)

    return polynomial
  
# Function to generate a set of plots with different parameters
def generate_plots(plot_function, original_function, parameter_list):
    num_plots = len(parameter_list)
    num_columns = 4  # Number of plots per line
    num_rows = (num_plots + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4*num_rows))
    axes = axes.ravel()  # Flatten the axes array for easy indexing

    for i, parameters in enumerate(parameter_list):
        ax = axes[i]
        plot_function(ax, original_function, **parameters)
        ax.set_title(f"a =  {parameters['interval']}, Degree = {parameters['degree']}")

    plt.tight_layout()
    plt.show()

def plot_poly(ax, original_function, degree, interval):
    x = np.linspace(-2, 2, 1000)
    polynomial = poly_approx(original_function, degree, 2, interval)
    ax.plot(x, original_function(x), label='Original Function')
    ax.plot(x, polynomial(x), label='Polynomial Approximation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim((-2, 2))
    ax.legend(loc = 'upper left', fontsize = 7)
    
def display_poly(function, polynomial, interval):
  x = np.linspace(-interval, interval, 1000)
  plt.plot(x, function(x), label='Original Step Function')
  plt.plot(x, polynomial(x), label = 'Polynomial Approximation')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()
