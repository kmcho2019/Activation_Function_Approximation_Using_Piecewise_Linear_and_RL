import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from itertools import accumulate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)

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

def silu_approx_10_pwl_paper(x):
    return x * sigmoid_approx_10_pwl_paper(x)

def matrix_row_generator(x, list_of_piecewise_points):
    vector = [1.] + [max(x - b, 0) for b in list_of_piecewise_points][:-1]
    return np.array(vector)

def piece_wise_linear_function_estimator(original_function, num_of_points, list_of_piecewise_points):
    min_x = list_of_piecewise_points[0]
    max_x = list_of_piecewise_points[-1]
    x = np.linspace(min_x, max_x, num_of_points)
    y = np.transpose(original_function(x))
    A = np.array([matrix_row_generator(x_i, list_of_piecewise_points) for x_i in x])
    piecewise_function_variables = []
    # print(x)
    # print(A)
    # print(A.shape)
    # print(A.T @ A)

    # Should not be used as A.T @ A can sometimes be singular and not invertible
    # beta = np.linalg.inv(A.T @ A) @ A.T @ y
    # print(beta)

    beta2 = scipy.linalg.lstsq(A, y)[0]
    # print(beta2)

    pseudo_inverse = np.linalg.pinv(A.T @ A)
    beta3 = pseudo_inverse @ A.T @ y
    # print(beta3)

    shift_variables = list(accumulate(beta2[1:]))
    for i in range(len(list_of_piecewise_points)-1):
        shift = shift_variables[i]
        bias = beta2[0] - sum([beta2[j+1] * list_of_piecewise_points[j] for j in range(i + 1)])
        piecewise_function_variables.append((shift, bias))
    # print('piecewise_function_variables')
    # print(piecewise_function_variables)
    return beta2, piecewise_function_variables


def piecewise_linear_function_constructor(piecewise_function_variables, list_of_piecewise_points):
    def piecewise_linear_function(x):
        for i in range(len(list_of_piecewise_points)-1):
            if x <= list_of_piecewise_points[i+1]:
                return piecewise_function_variables[i][0] * x + piecewise_function_variables[i][1]
    return piecewise_linear_function


# constructs a vectorized piecewise linear function
# piecewise_function_variables is a list of tuples of the form (slope, bias) length m
# list_of_piecewise_points is a list of points of length m + 1
# is_sigmoid is a boolean that determines whether the last function is a sigmoid or silu
def piecewise_linear_function_constructor_vectorized(piecewise_function_variables, list_of_piecewise_points, is_sigmoid=True):
    def piecewise_linear_function(x):
        conditions = []
        functions = []
        for i in range(len(list_of_piecewise_points)):
            if i == 0:
                functions.append(0.0)
                conditions.append(x <= list_of_piecewise_points[i])
            elif i == len(list_of_piecewise_points) - 1:
                conditions.append(np.logical_and(x > list_of_piecewise_points[i - 1], x <= list_of_piecewise_points[i]))
                conditions.append(x > list_of_piecewise_points[i])

                functions.append(
                    lambda x: piecewise_function_variables[i - 1][0] * x + piecewise_function_variables[i - 1][1])
                if is_sigmoid:
                    functions.append(1.0)
                else: # is silu
                    functions.append(lambda x: x)
            else:
                functions.append(lambda x, i=i: piecewise_function_variables[i-1][0] * x + piecewise_function_variables[i-1][1])
                conditions.append(np.logical_and(x > list_of_piecewise_points[i-1], x <= list_of_piecewise_points[i]))

        # conditions = np.logical_or.reduce(conditions)

        return np.piecewise(x, conditions, functions)

    return piecewise_linear_function


# combines the piecewise linear function estimator and the piecewise linear function constructor
def combined_function_generator(original_function, piecewise_points, number_of_points, is_sigmoid=True):
    beta, piecewise_function_variables = piece_wise_linear_function_estimator(original_function, number_of_points, piecewise_points)
    return piecewise_linear_function_constructor_vectorized(piecewise_function_variables, piecewise_points, is_sigmoid)

if __name__ == '__main__':

    x = np.linspace(-9, 9, 18000 + 1)

    conditions = []
    functions = []
    functions.append(0)
    functions.append(1)
    conditions.append(x < 0)
    conditions.append(x >= 0)


    test_y = np.piecewise(x, conditions, functions)

    plt.plot(x, test_y)
    plt.show()



    sigmoid_p4_points = [-8, -2, 2, 8]
    sigmoid_p4_beta2, sigmoid_p4_piecewise_function_variables = piece_wise_linear_function_estimator(sigmoid, 16000 + 1, sigmoid_p4_points)

    sigmoid_pwl_4 = piecewise_linear_function_constructor_vectorized(sigmoid_p4_piecewise_function_variables, sigmoid_p4_points)

    sigmoid_p6_points = [-8, -3, -1, 1, 3, 8]
    sigmoid_p6_beta2, sigmoid_p6_piecewise_function_variables = piece_wise_linear_function_estimator(sigmoid, 16000 + 1, sigmoid_p6_points)

    sigmoid_pwl_6 = piecewise_linear_function_constructor_vectorized(sigmoid_p6_piecewise_function_variables, sigmoid_p6_points)

    sigmoid_p8_points = [-8, -4, -2, -1, 1, 2, 4, 8]
    sigmoid_p8_beta2, sigmoid_p8_piecewise_function_variables = piece_wise_linear_function_estimator(sigmoid, 16000 + 1, sigmoid_p8_points)

    sigmoid_pwl_8 = piecewise_linear_function_constructor_vectorized(sigmoid_p8_piecewise_function_variables, sigmoid_p8_points)

    sigmoid_p10_points = [-8, -4.5, -3, -2, -1, 1, 2, 3, 4.5, 8]

    sigmoid_p10_beta2, sigmoid_p10_piecewise_function_variables = piece_wise_linear_function_estimator(sigmoid, 16000 + 1, sigmoid_p10_points)

    sigmoid_pwl_10 = piecewise_linear_function_constructor_vectorized(sigmoid_p10_piecewise_function_variables, sigmoid_p10_points)




    piece_wise_linear_function_estimator(sigmoid, 16000 + 1,[-8, -4.5, -3, -2.5, -2, -1.5, -1, 1, 1.5, 2, 2.5, 3, 4.5, 8])

    piece_wise_linear_function_estimator(silu, 16000 + 1,[-8, -4.5, -3, -2.5, -2, -1.5, -1, 1, 1.5, 2, 2.5, 3, 4.5, 8])

    # plot the piecewise linear functions
    plt.plot(x, sigmoid_pwl_4(x), label='sigmoid pwl 4')
    plt.plot(x, sigmoid_pwl_6(x), label='sigmoid pwl 6')
    plt.plot(x, sigmoid_pwl_8(x), label='sigmoid pwl 8')
    plt.plot(x, sigmoid_pwl_10(x), label='sigmoid pwl 10')
    plt.plot(x, sigmoid(x), label='sigmoid')
    plt.legend()
    plt.show()

    # do this for silu
    silu_p4_points = [-8, -2, 2, 8]
    silu_pwl_4 = combined_function_generator(silu, silu_p4_points, 16000 + 1, False)

    silu_p6_points = [-8, -3, -1, 1, 3, 8]
    silu_pwl_6 = combined_function_generator(silu, silu_p6_points, 16000 + 1, False)

    silu_p8_points = [-8, -4, -2, -1, 1, 2, 4, 8]
    silu_pwl_8 = combined_function_generator(silu, silu_p8_points, 16000 + 1, False)

    silu_p10_points = [-8, -4.5, -3, -2, -1, 1, 2, 3, 4.5, 8]
    silu_pwl_10 = combined_function_generator(silu, silu_p10_points, 16000 + 1, False)

    silu_p9_custom_points = [-8, -4, -1.5, -1, -0.5, 1, 1.5, 4, 8]
    silu_pwl_9_custom = combined_function_generator(silu, silu_p9_custom_points, 16000 + 1, False)

    silu_p11_custom_points = [-8, -4, -1.5, -1, -0.5, 0.0, 0.5, 1, 1.5, 4, 8]
    silu_pwl_11_custom = combined_function_generator(silu, silu_p11_custom_points, 16000 + 1, False)

    # plot the piecewise linear functions
    plt.plot(x, silu_pwl_4(x), label='silu pwl 4')
    plt.plot(x, silu_pwl_6(x), label='silu pwl 6')
    plt.plot(x, silu_pwl_8(x), label='silu pwl 8')
    plt.plot(x, silu_pwl_10(x), label='silu pwl 10')
    plt.plot(x, silu_pwl_9_custom(x), label='silu pwl 9 custom')
    plt.plot(x, silu(x), label='silu')
    plt.legend()
    plt.show()

    plt.plot(x, silu_pwl_8(x), label='silu pwl 8')
    plt.plot(x, silu_pwl_10(x), label='silu pwl 10')
    plt.plot(x, silu_pwl_9_custom(x), label='silu pwl 9 custom')
    plt.plot(x, silu(x), label='silu')
    plt.legend()
    plt.show()

    plt.rcParams["figure.figsize"] = (20, 10)
    plt.plot(x, silu_pwl_9_custom(x), label='silu pwl 9 custom')
    plt.plot(x, silu(x), label='silu')
    plt.xticks(np.arange(-9, 9.5, 0.5))
    plt.grid(True)
    plt.tight_layout()
    # make image a lot larger

    plt.legend()
    plt.show()

    plt.plot(x, silu_pwl_11_custom(x), label='silu pwl 11 custom')
    plt.plot(x, silu(x), label='silu')
    plt.legend()
    plt.show()


    plt.plot(x, silu_pwl_10(x), label='silu pwl 10')
    plt.plot(x, silu(x), label='silu')
    plt.legend()
    plt.show()

    plt.plot(x, silu_approx_10_pwl_paper(x), label='silu approx 10 pwl paper')
    plt.plot(x, silu(x), label='silu')
    plt.legend()
    plt.show()

    # compute error of silu approximations
    x_high_precision = np.linspace(-8, 8, 160000 + 1)
    silu_pwl_4_error = np.abs(silu_pwl_4(x_high_precision) - silu(x_high_precision))
    silu_pwl_6_error = np.abs(silu_pwl_6(x_high_precision) - silu(x_high_precision))
    silu_pwl_8_error = np.abs(silu_pwl_8(x_high_precision) - silu(x_high_precision))
    silu_pwl_10_error = np.abs(silu_pwl_10(x_high_precision) - silu(x_high_precision))
    silu_pwl_9_custom_error = np.abs(silu_pwl_9_custom(x_high_precision) - silu(x_high_precision))
    silu_pwl_11_custom_error = np.abs(silu_pwl_11_custom(x_high_precision) - silu(x_high_precision))
    sigmoid_approx_10_pwl_paper_error = np.abs(silu_approx_10_pwl_paper(x_high_precision) - silu(x_high_precision))


    #
    # silu_pwl_4_error = np.abs(silu_pwl_4(x) - silu(x))
    # silu_pwl_6_error = np.abs(silu_pwl_6(x) - silu(x))
    # silu_pwl_8_error = np.abs(silu_pwl_8(x) - silu(x))
    # silu_pwl_10_error = np.abs(silu_pwl_10(x) - silu(x))
    # silu_pwl_9_custom_error = np.abs(silu_pwl_9_custom(x) - silu(x))

    # plot the errors
    plt.plot(x_high_precision, silu_pwl_4_error, label='silu pwl 4 error')
    plt.plot(x_high_precision, silu_pwl_6_error, label='silu pwl 6 error')
    plt.plot(x_high_precision, silu_pwl_8_error, label='silu pwl 8 error')
    plt.plot(x_high_precision, silu_pwl_10_error, label='silu pwl 10 error')
    plt.plot(x_high_precision, silu_pwl_9_custom_error, label='silu pwl 9 custom error')
    plt.plot(x_high_precision, sigmoid_approx_10_pwl_paper_error, label='sigmoid approx 10 pwl paper error')
    plt.legend()
    plt.show()

    # plot the errors

    plt.plot(x_high_precision, silu_pwl_10_error, label='silu pwl 10 error')
    plt.plot(x_high_precision, silu_pwl_9_custom_error, label='silu pwl 9 custom error')
    plt.plot(x_high_precision, sigmoid_approx_10_pwl_paper_error, label='sigmoid approx 10 pwl paper error')
    plt.plot(x_high_precision, silu_pwl_11_custom_error, label='silu pwl 11 custom error')
    plt.xticks(np.arange(-8, 8.5, 0.5))
    plt.grid(True)
    plt.legend()
    plt.show()


    # print the max and average errors
    print('interval [-8, 8], 160001 points, interval 0.0001')
    print('silu pwl 4 max error: ', np.max(silu_pwl_4_error))
    print('silu pwl 4 average error: ', np.average(silu_pwl_4_error))
    print('silu pwl 6 max error: ', np.max(silu_pwl_6_error))
    print('silu pwl 6 average error: ', np.average(silu_pwl_6_error))
    print('silu pwl 8 max error: ', np.max(silu_pwl_8_error))
    print('silu pwl 8 average error: ', np.average(silu_pwl_8_error))
    print('silu pwl 10 max error: ', np.max(silu_pwl_10_error))
    print('silu pwl 10 average error: ', np.average(silu_pwl_10_error))
    print('silu pwl 9 custom max error: ', np.max(silu_pwl_9_custom_error))
    print('silu pwl 9 custom average error: ', np.average(silu_pwl_9_custom_error))
    print('silu pwl 11 custom max error: ', np.max(silu_pwl_11_custom_error))
    print('silu pwl 11 custom average error: ', np.average(silu_pwl_11_custom_error))
    print('silu pwl 10 sigmoid * x max error: ', np.max(sigmoid_approx_10_pwl_paper_error))
    print('silu pwl 10 sigmoid * x average error: ', np.average(sigmoid_approx_10_pwl_paper_error))

