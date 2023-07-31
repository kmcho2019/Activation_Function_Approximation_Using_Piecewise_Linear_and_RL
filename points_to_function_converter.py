# Able to take in a list of points and convert them to a piecewise linear approximation
import argparse
import numpy as np
from least_sq_approximation import combined_function_generator
from least_sq_approximation import piece_wise_linear_function_estimator
import matplotlib.pyplot as plt

# Define the functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def silu(x):
    return x / (1 + np.exp(-x))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


#Default values if no arguments are given
default_function = 'GELU'
default_points = [-5.4738750696182255, -2.5131248712539676, -0.7461250543594363, -0.3765439748764041, -0.007824301719665833, 0.35101768970489466, 0.7172322988510128, 2.5810040712356566]
default_range = [-8, 8]
default_round_decimals = 5



# Define the argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Function evaluation')
    parser.add_argument('-F', choices=['Sigmoid', 'SiLU', 'GELU'], default=default_function, help='Function to apply')
    parser.add_argument('-P', nargs='+', type=float, default=default_points, help='Points to evaluate, must be in ascending order')
    parser.add_argument('-R', nargs=2, type=float, default=default_range, help='Range (a and b)')
    parser.add_argument('--round', type=int, default=default_round_decimals, help='Number of decimal points to round the output')

    return parser.parse_args()

# Main function to evaluate the selected function on the given points within the range

def evaluate_function(args):
    min_range, max_range = min(args.R), max(args.R)

    if args.F == 'Sigmoid':
        selected_function = sigmoid
    elif args.F == 'SiLU':
        selected_function = silu
    elif args.F == 'GELU':
        selected_function = gelu

    print(f"Evaluating {args.F} function on points {args.P} within the range ({min_range}, {max_range}):")

    for point in args.P:
        if min_range <= point <= max_range:
            result = selected_function(point)
            print(f"{args.F}({point}) = {result:.4f}")
        else:
            print(f"Point {point} is out of the specified range.")
            raise argparse.ArgumentError(None, f"Point {point} is out of the specified range.")


# Print the function approximation and the variables
# Formats it in a way easy to understand
def print_function(args, piecewise_points, beta_variables, pwl_function_variables):
    if args.F == 'Sigmoid':
        M = 1
    else:
        M = 'x'
    print(f"{args.F} function approximation: ")
    print(f"betas: {beta_variables}")
    print(f"pwl_function variables (shift, bias): {pwl_function_variables}")
    print(f"f(x) = {{")
    for i in range(len(piecewise_points) - 1):
        if i == 0:
            print(f"        0, x <= {piecewise_points[0]}")
        if pwl_function_variables[i][1] < 0:
            print(f"        {pwl_function_variables[i][0]} * x - {-pwl_function_variables[i][1]}, {piecewise_points[i]} < x <= {piecewise_points[i + 1]}")
        else:
            print(f"        {pwl_function_variables[i][0]} * x + {pwl_function_variables[i][1]}, {piecewise_points[i]} < x <= {piecewise_points[i + 1]}")

    print(f"        {M}, x > {piecewise_points[-1]}")
    print(f"}}")
    # Print ordered betas
    for i in range(len(beta_variables)):
        print(f"beta_{i} = {beta_variables[i]}")
    # Print rounded betas and pwl_function variables
    print('\nRounded values: ')
    for i in range(len(beta_variables)):
        print(f"Rounded beta_{i} = {np.round(beta_variables[i], args.round)}")
    for i in range(len(pwl_function_variables)):
        print(f"Rounded pwl_function variable_{i} = {np.round(pwl_function_variables[i], args.round)}")

# Visualize the function approximation compared against the original function
# Slightly modified from version of chart_function_and_approximations in approximation_visualizer.py
def visualize_function(args, piecewise_points, pwl_function_variables, save_file_name=None):
    # Piecewise linear approximation function, used internally
    def piecewise_linear_approx(x, ranges, a, b):
        for i in range(len(ranges)):
            if ranges[i][0] <= x <= ranges[i][1]:
                return a[i] * x + b[i]
        return 0  # default case

    if args.F == 'Sigmoid':
        selected_function = sigmoid
    elif args.F == 'SiLU':
        selected_function = silu
    elif args.F == 'GELU':
        selected_function = gelu
    else: # should never happen, raise error
        raise Exception("Unknown function")

    function = selected_function
    function_name = args.F

    a_list = [ele[0] for ele in pwl_function_variables]
    b_list = [ele[1] for ele in pwl_function_variables]
    ranges = []
    for i in range(len(piecewise_points) - 1):
        ranges.append([piecewise_points[i], piecewise_points[i + 1]])

    # Plot function and its approximation with point marks for separating segments and coordinates
    range_minimum = ranges[0][0]
    range_maximum = ranges[-1][1]
    x = np.linspace(range_minimum, range_maximum, 1000)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
    fig, ax = plt.subplots(figsize=(12, 6))
    y = function(x)
    y_approx = [piecewise_linear_approx(i, ranges, a_list, b_list) for i in x]
    ax.plot(x, y, label=f"{function_name} function", linewidth=2)
    for i in range(len(ranges)):
        x_piecewise = np.linspace(ranges[i][0], ranges[i][1], 500)
        y_piecewise = [piecewise_linear_approx(j, [ranges[i]], [a_list[i]], [b_list[i]]) for j in x_piecewise]
        if b_list[i] < 0:
            function_label = f"{a_list[i]}x - {-b_list[i]} for ({ranges[i][0]}, {ranges[i][1]}]"
        else:
            function_label = f"{a_list[i]}x + {b_list[i]} for ({ranges[i][0]}, {ranges[i][1]}]"
        ax.plot(x_piecewise, y_piecewise, label=f"{function_label}",
                linestyle='dashed',
                color=colors[i])
        ax.plot(ranges[i][0],
                piecewise_linear_approx(ranges[i][0], [ranges[i]], [a_list[i]], [b_list[i]]), 'ko',
                markersize=4)  # lower bound
        ax.plot(ranges[i][1],
                piecewise_linear_approx(ranges[i][1], [ranges[i]], [a_list[i]], [b_list[i]]), 'ko',
                markersize=4)  # upper bound
        # adjust_text_position(ranges[i][0], piecewise_linear_approx(ranges[i][0], [ranges[i]], [a_list[i]], [b_list[i]]), f'({ranges[i][0]:.2f}, {piecewise_linear_approx(ranges[i][0], [ranges[i]], [a_list[i]], [b_list[i]]):.2f})', ax)  # lower bound coordinate
        # adjust_text_position(ranges[i][1], piecewise_linear_approx(ranges[i][1], [ranges[i]], [a_list[i]], [b_list[i]]), f'({ranges[i][1]:.2f}, {piecewise_linear_approx(ranges[i][1], [ranges[i]], [a_list[i]], [b_list[i]]):.2f})', ax)  # upper bound coordinate
    ax.set_title(f"{function_name} Function and Its Piecewise Linear Approximations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    if save_file_name is not None:
        fig.savefig(save_file_name, bbox_inches='tight', dpi=300)

def plot_error(args, pwl_function, piecewise_points,number_of_samples=100_000):
    if args.F == 'Sigmoid':
        selected_function = sigmoid
    elif args.F == 'SiLU':
        selected_function = silu
    elif args.F == 'GELU':
        selected_function = gelu
    else: # Raise Error, should never happen
        raise Exception("Unknown function")
    x = np.linspace(min(piecewise_points), max(piecewise_points), number_of_samples)
    y = selected_function(x)
    y_approx = pwl_function(x)
    absolute_error = np.abs(y - y_approx)
    plt.plot(x, absolute_error, label="absolute error", linewidth=2, color='darkgray')
    plt.title(f"Error of {args.F} function and its piecewise linear approximation")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    evaluate_function(args)
    if args.F == 'Sigmoid':
        is_sigmoid = True
    else:
        is_sigmoid = False
    if args.F == 'Sigmoid':
        selected_function = sigmoid
    elif args.F == 'SiLU':
        selected_function = silu
    elif args.F == 'GELU':
        selected_function = gelu
    else: # Raise Error, should never happen
        raise Exception("Unknown function")
    print(args)
    # combine the range and the points for the combined_function_generator
    piecewise_points = []
    piecewise_points.append(min(args.R))
    piecewise_points.extend(args.P)
    piecewise_points.append(max(args.R))
    print(piecewise_points)
    number_of_samples = 100_000
    beta_variables, pwl_function_variables = piece_wise_linear_function_estimator(selected_function, number_of_samples, piecewise_points)
    pwl_function = combined_function_generator(selected_function, piecewise_points, number_of_samples, is_sigmoid)
    print_function(args, piecewise_points, beta_variables, pwl_function_variables)
    visualize_function(args, piecewise_points, pwl_function_variables)
    plot_error(args, pwl_function, piecewise_points, number_of_samples)