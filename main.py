# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as sym
import math

def sech(x):
    return 1 / np.cosh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gelu_2_derivative(x):
    return 0.5 * np.tanh(0.0356774 * (x**3 + x)) + (0.0535161 * x**2 * sech(0.0356774 * (x**3 + x))) + 0.5


# full accuracy version of gelu function
def gelu_1(x):
    cdf = 0.5 * (1.0 + sp.special.erf(x / np.sqrt(2.0)))
    return x * cdf

# less precise version of gelu function using tanh
def gelu_2(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# even less precise version of gelu function using sigmoid
def gelu_3(x):
    return x * sigmoid(1.702 * x)

def silu_1(x):
    return x * sigmoid(x)

def silu_1_derivative(x):
    return sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

# approximation of silu function using polynomial, Taylor approximation
def silu_taylor_1(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=1, scale=1, order=1 + 2)
    return f(x)

def silu_taylor_1_derivative(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=1, scale=1, order=1 + 2)
    f = f.deriv(m=1)
    return f(x)

def silu_taylor_2(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=2, scale=1, order=2 + 2)
    return f(x)

def silu_taylor_2_derivative(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=2, scale=1, order=2 + 2)
    f = f.deriv(m=1)
    return f(x)

def silu_taylor_3(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=3, scale=1, order=3 + 2)
    return f(x)

def silu_taylor_3_derivative(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=3, scale=1, order=3 + 2)
    f = f.deriv(m=1)
    return f(x)

def silu_taylor_4(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=4, scale=1, order=4 + 2)
    return f(x)

def silu_taylor_4_derivative(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=4, scale=1, order=4 + 2)
    f = f.deriv(m=1)
    return f(x)

def silu_taylor_10(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=10, scale=1, order=10 + 2)
    return f(x)

def silu_taylor_10_derivative(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=10, scale=1, order=10 + 2)
    f = f.deriv(m=1)
    return f(x)

def silu_taylor_20(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=20, scale=1, order=20 + 2)
    return f(x)

def silu_taylor_20_derivative(x):
    f = sp.interpolate.approximate_taylor_polynomial(silu_1, 0, degree=20, scale=1, order=20 + 2)
    f = f.deriv(m=1)
    return f(x)

def main():
    x = np.linspace(-5, 5, 1000)
    y1 = gelu_1(x)
    y2 = gelu_2(x)
    y3 = gelu_3(x)

    # plot the gelu function and its approximations
    plt.plot(x, y1, label='GELU')
    plt.plot(x, y2, label='Approximation using tanh')
    plt.plot(x, y3, label='Approximation using sigmoid')
    plt.legend()
    plt.show()
    # second diagram that compares the error of gelu_2 and gelu_3 against gelu_1
    y2_error = np.abs(y2 - y1)
    y3_error = np.abs(y3 - y1)
    plt.plot(x, y2_error, label='Error of tanh approximation')
    plt.plot(x, y3_error, label='Error of sigmoid approximation')
    plt.legend()
    plt.show()

    x_long = np.linspace(-10, 10, 10000, dtype=np.float64)
    y4 = gelu_2_derivative(x_long)
    # plot the derivative of gelu function and its approximation
    plt.plot(x_long, y4, label='Derivative of tanh approximation')
    plt.legend()
    plt.show()

    # approximations for silu function
    x = np.linspace(-3, 3, 1000)

    y5 = silu_1(x)
    y6 = silu_taylor_1(x)
    y7 = silu_taylor_2(x)
    y8 = silu_taylor_3(x)
    y9 = silu_taylor_4(x)
    y10 = silu_taylor_10(x)
    y11 = silu_taylor_20(x)


    # plot the silu function and its approximations
    plt.plot(x, y5, label='SiLU')
    plt.plot(x, y6, label='Approximation using Taylor polynomial of degree 1')
    plt.plot(x, y7, label='Approximation using Taylor polynomial of degree 2')
    plt.plot(x, y8, label='Approximation using Taylor polynomial of degree 3')
    plt.plot(x, y9, label='Approximation using Taylor polynomial of degree 4')
    plt.plot(x, y10, label='Approximation using Taylor polynomial of degree 10')
    plt.plot(x, y11, label='Approximation using Taylor polynomial of degree 20')

    plt.legend()
    plt.show()

    # second diagram that compares the error of silu approximations against silu_1

    y6_error = np.abs(y6 - y5)
    y7_error = np.abs(y7 - y5)
    y8_error = np.abs(y8 - y5)
    y9_error = np.abs(y9 - y5)
    y10_error = np.abs(y10 - y5)
    y11_error = np.abs(y11 - y5)
    plt.plot(x, y6_error, label='Error of Taylor polynomial of degree 1')
    plt.plot(x, y7_error, label='Error of Taylor polynomial of degree 2')
    plt.plot(x, y8_error, label='Error of Taylor polynomial of degree 3')
    plt.plot(x, y9_error, label='Error of Taylor polynomial of degree 4')
    plt.plot(x, y10_error, label='Error of Taylor polynomial of degree 10')
    plt.plot(x, y11_error, label='Error of Taylor polynomial of degree 20')
    # make y axis error be log
    # plt.yscale('log')

    plt.legend()
    plt.show()
    # plot the derivative of silu function and its approximation
    y5_derv = silu_1_derivative(x)
    y6_derv = silu_taylor_1_derivative(x)
    y7_derv = silu_taylor_2_derivative(x)
    y8_derv = silu_taylor_3_derivative(x)
    y9_derv = silu_taylor_4_derivative(x)
    y10_derv = silu_taylor_10_derivative(x)
    y11_derv = silu_taylor_20_derivative(x)
    plt.plot(x, y5_derv, label='Derivative of SiLU')
    plt.plot(x, y6_derv, label='Approximation using Taylor polynomial of degree 1')
    plt.plot(x, y7_derv, label='Approximation using Taylor polynomial of degree 2')
    plt.plot(x, y8_derv, label='Approximation using Taylor polynomial of degree 3')
    plt.plot(x, y9_derv, label='Approximation using Taylor polynomial of degree 4')
    plt.plot(x, y10_derv, label='Approximation using Taylor polynomial of degree 10')
    plt.plot(x, y11_derv, label='Approximation using Taylor polynomial of degree 20')
    plt.legend()
    plt.show()
    print("hello world")
    print('SiLU at x = 5', silu_1(5))
    print('SiLU derivative at x = 5', silu_1_derivative(5))
    print('SiLU at x = 10', silu_1(10))
    print('SiLU derivative at x = 10', silu_1_derivative(10))
    print('SiLU at x = -5', silu_1(-5))
    print('SiLU derivative at x = -5', silu_1_derivative(-5))
    print('SiLU at x = -10', silu_1(-10))
    print('SiLU derivative at x = -10', silu_1_derivative(-10))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
