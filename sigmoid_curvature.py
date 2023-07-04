import numpy as np
import scipy
import matplotlib.pyplot as plt
import math

import numpy as np
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from scipy.linalg import lstsq



def relu6(x):
    return np.minimum(np.maximum(0, x), 6)

def h_swish(x):
    return x * relu6(x + 3) / 6


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_curvature(x):
    return (np.abs(np.exp(-2 * x)-np.exp(-x))* np.power(1+np.exp(-x),3))/np.sqrt(np.power(1 + 4 * np.exp(-x) + 7 * np.exp(-2 * x) + 4 * np.exp(-3 * x) + np.exp(-4 * x), 3))

def sigmoid_curvature_alt(x):
    return (16 * np.power(np.cosh(x / 2), 3) * abs(np.sinh(x / 2))) / np.sqrt(np.power(16 * np.power(np.cosh(x / 2), 4) + 1,3))

# sigmoid approximation function proposed by paper:
# FPGA Implementation for the Sigmoid Function Using Piecewise Linear Approximation Fitting Method Based on Curvature Analysis
# https://www.mdpi.com/2079-9292/11/9/1365
def sigmoid_approx_8_pwl_paper(x):
    y = np.zeros_like(x)  # Create an array of zeros with the same shape as x

    # Define logical conditions for each range
    condition1 = x <= -8
    condition2 = np.logical_and(-8 < x, x <= -4)
    condition3 = np.logical_and(-4 < x, x <= -2)
    condition4 = np.logical_and(-2 < x, x <= -1)
    condition5 = np.logical_and(-1 < x, x <= 1)
    condition6 = np.logical_and(1 < x, x <= 2)
    condition7 = np.logical_and(2 < x, x <= 4)
    condition8 = np.logical_and(4 < x, x <= 8)
    condition9 = x > 8

    # Apply the piecewise linear approximation using the conditions
    y[condition1] = 0.
    y[condition2] = 0.00261 * x[condition2] + 0.01947
    y[condition3] = 0.04767 * x[condition3] + 0.19971
    y[condition4] = 0.15881 * x[condition4] + 0.42199
    y[condition5] = 0.23682 * x[condition5] + 0.5
    y[condition6] = 0.15881 * x[condition6] + 0.57801
    y[condition7] = 0.04767 * x[condition7] + 0.80029
    y[condition8] = 0.00261 * x[condition8] + 0.98053
    y[condition9] = 1.

    return y

def sigmoid_approx_10_pwl_paper(x):
    y = np.zeros_like(x)  # Create an array of zeros with the same shape as x

    # Define logical conditions for each range
    condition1 = x <= -8
    condition2 = np.logical_and(-8 < x, x <= -4.5)
    condition3 = np.logical_and(-4.5 < x, x <= -3)
    condition4 = np.logical_and(-3 < x, x <= -2)
    condition5 = np.logical_and(-2 < x, x <= -1)
    condition6 = np.logical_and(-1 < x, x <= 1)
    condition7 = np.logical_and(1 < x, x <= 2)
    condition8 = np.logical_and(2 < x, x <= 3)
    condition9 = np.logical_and(3 < x, x <= 4.5)
    condition10 = np.logical_and(4.5 < x, x <= 8)
    condition11 = x > 8

    # Apply the piecewise linear approximation using the conditions
    y[condition1] = 0.
    y[condition2] = 0.00252 * x[condition2] + 0.01875
    y[condition3] = 0.02367 * x[condition3] + 0.11397
    y[condition4] = 0.06975 * x[condition4] + 0.25219
    y[condition5] = 0.14841 * x[condition5] + 0.40951
    y[condition6] = 0.2389 * x[condition6] + 0.5
    y[condition7] = 0.1481 * x[condition7] + 0.59049
    y[condition8] = 0.06975 * x[condition8] + 0.74781
    y[condition9] = 0.02367 * x[condition9] + 0.88603
    y[condition10] = 0.00252 * x[condition10] + 0.98125
    y[condition11] = 1.

    return y

def sigmoid_approx_8_pwl_non_vectorized_paper(x): # only able to return single value
    if x <= -8:
        return 0.
    elif -8 < x <= -4:
        return 0.00261 * x + 0.01947
    elif -4 < x <= -2:
        return 0.04767 * x + 0.19971
    elif -2 < x <= -1:
        return 0.15881 * x + 0.42199
    elif -1 < x <= 1:
        return 0.23682 * x + 0.5
    elif 1 < x <= 2:
        return 0.15881 * x + 0.57801
    elif 2 < x <= 4:
        return 0.04767 * x + 0.80029
    elif 4 < x <= 8:
        return 0.00261 * x + 0.98053
    else:
        return 1.

def silu(x):
    return x * sigmoid(x)

def silu_derivative(x):
    return sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

def silu_double_derivative(x):
    return sigmoid(x) * (1- sigmoid(x)) + sigmoid(x) * (1- sigmoid(x))  + x * sigmoid(x) * np.power(1 - sigmoid(x),2) - x * np.power(sigmoid(x), 2) * (1 - sigmoid(x))

def silu_curvature(x):
    return np.abs(silu_double_derivative(x)) / np.power(1 + np.power(silu_derivative(x), 2), 1.5)

def silu_curvature_alt(x):
    return (2 * np.power(np.cosh(x) + 1,3) * np.abs((np.exp(x) + 1) * np.cosh(x/2) - (x + np.exp(x) + 1) * np.sinh(x/2))) / (np.power(np.power(x + np.exp(x) + 1,2) + 16 * np.power(np.cosh(x/2), 4), 1.5) * np.power(np.cosh(x/2), 3))

# generate silu approximation based on paper sigmoid implementation
# based on 8 piecewise linear approximation version
def silu_approx_8_pwl_paper(x):
    return x * sigmoid_approx_8_pwl_paper(x)

# based on 10 piecewise linear approximation version
def silu_approx_10_pwl_paper(x):
    return x * sigmoid_approx_10_pwl_paper(x)


def sigmoid_curvature_dbscan_clustering():
    # Set the desired number of points
    total_points = 10

    # Generate function values for the given range
    x_values = np.linspace(-8, 8, 1000)
    function_values = sigmoid_curvature(x_values)

    # Reshape the function values for DBSCAN input
    X = function_values.reshape(-1, 1)

    # Apply DBSCAN
    epsilon = 0.1  # Distance threshold for neighborhood definition
    min_samples = total_points  # Use total_points as min_samples for DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(X)

    # Get the cluster labels
    labels = dbscan.labels_

    # Allocate a fixed number of points based on the clusters
    allocated_points = []
    for cluster_label in np.unique(labels):
        cluster_points = x_values[labels == cluster_label]
        num_cluster_points = min(total_points // len(np.unique(labels)), len(cluster_points))
        allocated_points.extend(np.random.choice(cluster_points, num_cluster_points, replace=False))

    allocated_points = np.array(allocated_points)  # Convert to NumPy array

    # Plot the function values and allocated points
    plt.plot(x_values, function_values, label='Sigmoid Curvature')
    plt.scatter(allocated_points, sigmoid_curvature(allocated_points), color='red', label='Allocated Points')
    plt.xlabel('x')
    plt.ylabel('Sigmoid Curvature')
    plt.title('Point Allocation using DBSCAN')
    plt.legend()
    plt.show()

def silu_curvature_dbscan_clustering():
    # Set the desired number of points
    total_points = 10

    # Generate function values for the given range
    x_values = np.linspace(-8, 8, 1000)
    function_values = silu_curvature(x_values)

    # Reshape the function values for DBSCAN input
    X = function_values.reshape(-1, 1)

    # Apply DBSCAN
    epsilon = 0.1  # Distance threshold for neighborhood definition
    min_samples = total_points  # Use total_points as min_samples for DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(X)

    # Get the cluster labels
    labels = dbscan.labels_

    # Allocate a fixed number of points based on the clusters
    allocated_points = []
    for cluster_label in np.unique(labels):
        cluster_points = x_values[labels == cluster_label]
        num_cluster_points = min(total_points // len(np.unique(labels)), len(cluster_points))
        allocated_points.extend(np.random.choice(cluster_points, num_cluster_points, replace=False))

    allocated_points = np.array(allocated_points)  # Convert to NumPy array

    # Plot the function values and allocated points
    plt.plot(x_values, function_values, label='SiLU Curvature')
    plt.scatter(allocated_points, silu_curvature(allocated_points), color='red', label='Allocated Points')
    plt.xlabel('x')
    plt.ylabel('SiLU Curvature')
    plt.title('Point Allocation using DBSCAN')
    plt.legend()
    plt.show()

def least_square_regression(func, range_min, range_max, seg_point_list):
    num_points = len(seg_point_list)
    x = np.linspace(range_min, range_max, num_points)
    y = func(x)


    return 0

def test():
    # Define the function you want to fit
    def my_function(x):
        return np.sin(x)  # Replace this with your desired function

    my_function = sigmoid
    # Set the range and number of points
    a = -8#0
    b = 8 #np.pi
    num_points = 10

    # Generate the x-values for the points
    x_values = np.linspace(a, b, num_points)

    # Evaluate the function at each x-value
    y_values = my_function(x_values)

    # Create the design matrix
    X = np.column_stack((x_values, np.ones_like(x_values)))

    # Perform piecewise linear fitting using OLS
    coefficients, _, _, _ = lstsq(X, y_values)

    # Compute the predicted y-values
    y_predicted = np.dot(X, coefficients)

    # Plot the original function and the piecewise linear fit
    x = np.linspace(a, b, 1000)
    y = my_function(x)
    plt.plot(x, y, label='Original Function')
    plt.plot(x_values, y_predicted, 'r--', label='Piecewise Linear Fit')
    plt.scatter(x_values, y_values, color='black', label='Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Piecewise Linear Fit using OLS')
    plt.legend()
    plt.show()

def main():
    x = np.linspace(-10, 10, 10000)
    y1 = sigmoid(x)
    y2 = sigmoid_derivative(x)
    y3 = sigmoid_curvature(x)
    y4 = sigmoid_curvature_alt(x)
    plt.plot(x, y1, label='sigmoid')
    plt.plot(x, y2, label='sigmoid derivative')
    plt.plot(x, y3, label='sigmoid curvature')
    # plt.plot(x, y4, label='sigmoid curvature alt')
    plt.legend()
    plt.show()

    # compare sigmoid_curvature and sigmoid_curvature_alt
    curvature_diff = np.abs(y3 - y4)
    plt.plot(x, curvature_diff, label='curvature difference')
    plt.legend()
    plt.show()
    average_curvature_diff = np.mean(curvature_diff)
    max_curvature_diff = np.max(curvature_diff)
    print('sigmoid average curvature difference: ', average_curvature_diff)
    print('sigmoid max curvature difference: ', max_curvature_diff)

    # silu
    y5 = silu(x)
    y6 = silu_derivative(x)
    y7 = silu_curvature(x)
    y8 = silu_curvature_alt(x)
    plt.plot(x, y5, label='silu')
    plt.plot(x, y6, label='silu derivative')
    plt.plot(x, y7, label='silu curvature')
    # plt.plot(x, y8, label='silu curvature alt')
    plt.legend()
    plt.show()

    # compare silu_curvature and silu_curvature_alt
    curvature_diff = np.abs(y7 - y8)
    plt.plot(x, curvature_diff, label='curvature difference')
    plt.legend()
    plt.show()
    average_curvature_diff = np.mean(curvature_diff)
    max_curvature_diff = np.max(curvature_diff)
    print('silu average curvature difference: ', average_curvature_diff)
    print('silu max curvature difference: ', max_curvature_diff)

    print(silu(8))
    # sigmoid_curvature_dbscan_clustering()
    # silu_curvature_dbscan_clustering()
    #
    # test()

    # compare sigmoid and sigmoid_approx_8_pwl (proposed in paper)
    paper_x = np.linspace(-8, 8, 1600)
    paper_y_sigmoid = sigmoid(paper_x)
    paper_y_approx_8_pwl_sigmoid = sigmoid_approx_8_pwl_paper(paper_x)
    paper_y_approx_10_pwl_sigmoid = sigmoid_approx_10_pwl_paper(paper_x)

    approx_average_error_8_pwl = np.mean(np.abs(paper_y_sigmoid - paper_y_approx_8_pwl_sigmoid))
    approx_max_error_8_pwl = np.max(np.abs(paper_y_sigmoid - paper_y_approx_8_pwl_sigmoid))
    approx_average_error_10_pwl = np.mean(np.abs(paper_y_sigmoid - paper_y_approx_10_pwl_sigmoid))
    approx_max_error_10_pwl = np.max(np.abs(paper_y_sigmoid - paper_y_approx_10_pwl_sigmoid))

    plt.plot(paper_x, paper_y_sigmoid, label='sigmoid')
    plt.plot(paper_x, paper_y_approx_8_pwl_sigmoid, label='sigmoid approx 8 pwl, paper')
    plt.plot(paper_x, paper_y_approx_10_pwl_sigmoid, label='sigmoid approx 10 pwl, paper')
    plt.legend()
    plt.show()

    plt.plot(paper_x, paper_y_sigmoid, label='sigmoid')
    plt.plot(paper_x, paper_y_approx_8_pwl_sigmoid, label='sigmoid approx 8 pwl, paper')
    plt.legend()
    plt.show()


    plt.plot(paper_x, paper_y_sigmoid, label='sigmoid')
    plt.plot(paper_x, paper_y_approx_10_pwl_sigmoid, label='sigmoid approx 10 pwl, paper')
    plt.legend()
    plt.show()


    print('interval [-8, 8], 1600 points')
    print('sigmoid approx 8 pwl, paper average error: ', approx_average_error_8_pwl)
    print('sigmoid approx 8 pwl, paper max error: ', approx_max_error_8_pwl)
    print('sigmoid approx 10 pwl, paper average error: ', approx_average_error_10_pwl)
    print('sigmoid approx 10 pwl, paper max error: ', approx_max_error_10_pwl)


    x_high_precision = np.linspace(-8, 8, 160000)
    paper_y_sigmoid = sigmoid(x_high_precision)
    paper_y_approx_8_pwl_sigmoid = sigmoid_approx_8_pwl_paper(x_high_precision)
    paper_y_approx_10_pwl_sigmoid = sigmoid_approx_10_pwl_paper(x_high_precision)

    approx_average_error_8_pwl = np.mean(np.abs(paper_y_sigmoid - paper_y_approx_8_pwl_sigmoid))
    approx_max_error_8_pwl = np.max(np.abs(paper_y_sigmoid - paper_y_approx_8_pwl_sigmoid))
    approx_average_error_10_pwl = np.mean(np.abs(paper_y_sigmoid - paper_y_approx_10_pwl_sigmoid))
    approx_max_error_10_pwl = np.max(np.abs(paper_y_sigmoid - paper_y_approx_10_pwl_sigmoid))

    plt.plot(x_high_precision, paper_y_sigmoid, label='sigmoid')
    plt.plot(x_high_precision, paper_y_approx_8_pwl_sigmoid, label='sigmoid approx 8 pwl, paper')
    plt.plot(x_high_precision, paper_y_approx_10_pwl_sigmoid, label='sigmoid approx 10 pwl, paper')
    plt.legend()
    plt.show()

    plt.plot(x_high_precision, paper_y_sigmoid, label='sigmoid')
    plt.plot(x_high_precision, paper_y_approx_8_pwl_sigmoid, label='sigmoid approx 8 pwl, paper')
    plt.legend()
    plt.show()


    plt.plot(x_high_precision, paper_y_sigmoid, label='sigmoid')
    plt.plot(x_high_precision, paper_y_approx_10_pwl_sigmoid, label='sigmoid approx 10 pwl, paper')
    plt.legend()
    plt.show()


    print('interval [-8, 8], 160000 points')
    print('sigmoid approx 8 pwl, paper average error: ', approx_average_error_8_pwl)
    print('sigmoid approx 8 pwl, paper max error: ', approx_max_error_8_pwl)
    print('sigmoid approx 10 pwl, paper average error: ', approx_average_error_10_pwl)
    print('sigmoid approx 10 pwl, paper max error: ', approx_max_error_10_pwl)

    # compare silu and silu_approx_8_pwl (proposed in paper)
    paper_x = np.linspace(-8, 8, 1600)
    paper_y_silu = silu(paper_x)
    paper_y_approx_8_pwl_silu = silu_approx_8_pwl_paper(paper_x)
    paper_y_approx_10_pwl_silu = silu_approx_10_pwl_paper(paper_x)

    approx_average_error_8_pwl = np.mean(np.abs(paper_y_silu - paper_y_approx_8_pwl_silu))
    approx_max_error_8_pwl = np.max(np.abs(paper_y_silu - paper_y_approx_8_pwl_silu))
    approx_average_error_10_pwl = np.mean(np.abs(paper_y_silu - paper_y_approx_10_pwl_silu))
    approx_max_error_10_pwl = np.max(np.abs(paper_y_silu - paper_y_approx_10_pwl_silu))

    plt.plot(paper_x, paper_y_silu, label='SiLU')
    plt.plot(paper_x, paper_y_approx_8_pwl_silu, label='SiLU approx 8 pwl, paper')
    plt.plot(paper_x, paper_y_approx_10_pwl_silu, label='SiLU approx 10 pwl, paper')
    plt.legend()
    plt.show()

    plt.plot(paper_x, paper_y_silu, label='SiLU')
    plt.plot(paper_x, paper_y_approx_8_pwl_silu, label='SiLU approx 8 pwl, paper')
    plt.legend()
    plt.show()


    plt.plot(paper_x, paper_y_silu, label='SiLU')
    plt.plot(paper_x, paper_y_approx_10_pwl_silu, label='SiLU approx 10 pwl, paper')
    plt.legend()
    plt.show()


    print('interval [-8, 8], 1600 points')
    print('SiLU approx 8 pwl, paper average error: ', approx_average_error_8_pwl)
    print('SiLU approx 8 pwl, paper max error: ', approx_max_error_8_pwl)
    print('SiLU approx 10 pwl, paper average error: ', approx_average_error_10_pwl)
    print('SiLU approx 10 pwl, paper max error: ', approx_max_error_10_pwl)

    # compare silu and silu_approx_8_pwl (proposed in paper)
    x_high_precision = np.linspace(-8, 8, 160000)
    paper_y_silu = silu(x_high_precision)
    paper_y_approx_8_pwl_silu = silu_approx_8_pwl_paper(x_high_precision)
    paper_y_approx_10_pwl_silu = silu_approx_10_pwl_paper(x_high_precision)

    approx_average_error_8_pwl = np.mean(np.abs(paper_y_silu - paper_y_approx_8_pwl_silu))
    approx_max_error_8_pwl = np.max(np.abs(paper_y_silu - paper_y_approx_8_pwl_silu))
    approx_average_error_10_pwl = np.mean(np.abs(paper_y_silu - paper_y_approx_10_pwl_silu))
    approx_max_error_10_pwl = np.max(np.abs(paper_y_silu - paper_y_approx_10_pwl_silu))

    plt.plot(x_high_precision, paper_y_silu, label='SiLU')
    plt.plot(x_high_precision, paper_y_approx_8_pwl_silu, label='SiLU approx 8 pwl, paper')
    plt.plot(x_high_precision, paper_y_approx_10_pwl_silu, label='SiLU approx 10 pwl, paper')
    plt.legend()
    plt.show()

    plt.plot(x_high_precision, paper_y_silu, label='SiLU')
    plt.plot(x_high_precision, paper_y_approx_8_pwl_silu, label='SiLU approx 8 pwl, paper')
    plt.legend()
    plt.show()


    plt.plot(x_high_precision, paper_y_silu, label='SiLU')
    plt.plot(x_high_precision, paper_y_approx_10_pwl_silu, label='SiLU approx 10 pwl, paper')
    plt.legend()
    plt.show()


    print('interval [-8, 8], 160000 points')
    print('SiLU approx 8 pwl, paper average error: ', approx_average_error_8_pwl)
    print('SiLU approx 8 pwl, paper max error: ', approx_max_error_8_pwl)
    print('SiLU approx 10 pwl, paper average error: ', approx_average_error_10_pwl)
    print('SiLU approx 10 pwl, paper max error: ', approx_max_error_10_pwl)

    # visualize sigmoid and silu curvature
    x = np.linspace(-8, 8, 1600)
    sigmoid_curve = sigmoid_curvature(x)
    silu_curve = silu_curvature(x)
    plt.plot(x, sigmoid_curve, label='sigmoid curvature')
    plt.plot(x, silu_curve, label='SiLU curvature')
    # Add lines at specific x-values
    x_lines = [-4.5, -3, -2, -1, 1, 2, 3, 4.5]
    for x_line in x_lines:
        plt.axvline(x=x_line, color='red', linestyle='--')
    plt.legend()
    plt.show()

    # visualize sigmoid and silu curvature
    x = np.linspace(-8, 8, 1600)
    sigmoid_curve = sigmoid_curvature(x)
    silu_curve = silu_curvature(x)
    plt.plot(x, sigmoid_curve, label='sigmoid curvature')
    plt.plot(x, silu_curve, label='SiLU curvature')

    plt.legend()
    plt.show()






























































    print('hello world')

if __name__ == '__main__':
    main()