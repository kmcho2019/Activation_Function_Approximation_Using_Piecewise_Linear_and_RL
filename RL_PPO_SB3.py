import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import datetime
import os
import tqdm
import time
from scipy.optimize import fsolve
from typing import Callable


from least_sq_approximation import matrix_row_generator
from least_sq_approximation import piece_wise_linear_function_estimator
from least_sq_approximation import piecewise_linear_function_constructor
from least_sq_approximation import piecewise_linear_function_constructor_vectorized
from least_sq_approximation import combined_function_generator



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def sigmoid_curvature(x):
    return (np.abs(np.exp(-2 * x)-np.exp(-x))* np.power(1+np.exp(-x),3))/np.sqrt(np.power(1 + 4 * np.exp(-x) + 7 * np.exp(-2 * x) + 4 * np.exp(-3 * x) + np.exp(-4 * x), 3))

def sigmoid_curvature_alt(x):
    return (16 * np.power(np.cosh(x / 2), 3) * abs(np.sinh(x / 2))) / np.sqrt(np.power(16 * np.power(np.cosh(x / 2), 4) + 1,3))

def silu_derivative(x):
    return sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

def silu_double_derivative(x):
    return sigmoid(x) * (1- sigmoid(x)) + sigmoid(x) * (1- sigmoid(x))  + x * sigmoid(x) * np.power(1 - sigmoid(x),2) - x * np.power(sigmoid(x), 2) * (1 - sigmoid(x))

def silu_curvature(x):
    return np.abs(silu_double_derivative(x)) / np.power(1 + np.power(silu_derivative(x), 2), 1.5)

def silu_curvature_alt(x):
    return (2 * np.power(np.cosh(x) + 1,3) * np.abs((np.exp(x) + 1) * np.cosh(x/2) - (x + np.exp(x) + 1) * np.sinh(x/2))) / (np.power(np.power(x + np.exp(x) + 1,2) + 16 * np.power(np.cosh(x/2), 4), 1.5) * np.power(np.cosh(x/2), 3))

def gelu_curvature(x):
    # Constants
    a = 0.044715
    sqrt_2_pi = np.sqrt(2 / np.pi)

    # First derivative
    y1 = 0.3989422804014327 * x * (1 + 0.134145 * x ** 2) * np.cosh(sqrt_2_pi * (x + a * x ** 3)) ** (-2) + 0.5 * (
                1 + np.tanh(sqrt_2_pi * (x + a * x ** 3)))

    # Second derivative
    y2 = 0.7978845608028654 * (1 + 0.134145 * x ** 2) * np.cosh(sqrt_2_pi * (x + a * x ** 3)) ** (-2) + 0.5 * x * (
                0.21406444881780073 * x * np.cosh(sqrt_2_pi * (x + a * x ** 3)) ** (-2) - (
                    4 * (1 + 0.134145 * x ** 2) ** 2 * np.cosh(sqrt_2_pi * (x + a * x ** 3)) ** (-2) * np.tanh(
                sqrt_2_pi * (x + a * x ** 3))) / np.pi)

    # Curvature
    curvature = np.abs(y2) / ((1 + y1 ** 2) ** (3 / 2))

    return curvature


# serves as the common reward function for final_reward_function_silu and final_reward_function_silu_print
def common_reward_function(mean_error, max_error):
    weighted_error = 0.2 * mean_error + 0.8 * max_error
    reward = 100 * np.exp(-weighted_error * 5)
    return reward

# new reward function to deal with the issues of the previous common_reward_function
# the previous reward function was too generous with errors that were relatively large
# for example mean error: 0.015337637732471146, max error: 0.08115555668721605
# resulted in error of 71.17988906069726, with maximum possible reward being 100
# this is fairly good result respect to the maximum but the error is still relatively large
# RL model had a problem with this and got stuck on a local minima and could not improve
# (this was especially prone with the gelu function but happened with silu as well)
# points tended to cluster around and skip small nuances of the function
# this reward function is stricter with errors that are relatively large and has a sharper curve respect to input errors
# rewards increase much more quickly with smaller errors and vice versa
# for example the above example error(mean:0.015337637732471146, max: 0.08115555668721605)
# results in error of 14.686018886887354, with maximum possible reward being 10000 (when approximation has 0 error)
# this encourages the model to explore more and not get stuck on local minima, due to promise of greater reward
# reasonable expectation for best error is around 0.001 which would result in reward of 1000
def reciprocal_error_average_reward_function(mean_error, max_error):
    # add a small constant to avoid division by zero and cap the reward at 10000
    weighted_error = 0.2 * mean_error + 0.8 * max_error + 0.0001
    reward = 1 / weighted_error
    return reward


# takes in the list of all points chosen by the agent and the initial range
# then it constructs a piecewise linear approximation of the silu function
# it computes the mean and max error of the approximation compared to the silu function
# with the mean and max error it constructs a reward function that is used by the agent
def final_reward_function_silu(chosen_points, initial_range, input_reward_function=common_reward_function):
    # append the min value of the range and the max value of the range to the first and last position of the list
    # with chosen_points filling the middle
    # this is needed as the piecewise linear function constructor needs the first and last point (the range) to be the min and max
    piecewise_function_segments = []
    piecewise_function_segments.append(initial_range[0])
    piecewise_function_segments.extend(chosen_points)
    if chosen_points[-1] != initial_range[1]:
        piecewise_function_segments.append(initial_range[1])
    step_size = 0.001 # determines the accuracy of the least square fit that generates the approximation of the silu function
    total_number_of_steps = int((initial_range[1] - initial_range[0]) / step_size) + 1
    # construct the piecewise linear approximation of the silu function
    piecewise_function = combined_function_generator(silu, piecewise_function_segments, total_number_of_steps, False)
    # compute the mean and max error of the approximation compared to the silu function
    x = np.linspace(initial_range[0], initial_range[1], total_number_of_steps)
    silu_reference_val = silu(x)
    silu_approximation_val = piecewise_function(x)
    mean_error = np.mean(np.abs(silu_reference_val - silu_approximation_val))
    max_error = np.max(np.abs(silu_reference_val - silu_approximation_val))
    # construct the reward function
    # the reward function takes the linear weighted sum of mean and max error and takes a form of a
    # monotonic decreasing function whose value is between 0 and 10
    # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
    reward = input_reward_function(mean_error, max_error)

    return reward

# takes in the list of all points chosen by the agent and the initial range
# and prints the values of the reward and error
def final_reward_function_silu_print(chosen_points, initial_range, verbose = True, input_reward_function=common_reward_function):
    # append the min value of the range and the max value of the range to the first and last position of the list
    # with chosen_points filling the middle
    # this is needed as the piecewise linear function constructor needs the first and last point (the range) to be the min and max
    piecewise_function_segments = []
    piecewise_function_segments.append(initial_range[0])
    piecewise_function_segments.extend(chosen_points)
    if chosen_points[-1] != initial_range[1]:
        piecewise_function_segments.append(initial_range[1])
    step_size = 0.001 # determines the accuracy of the least square fit that generates the approximation of the silu function
    total_number_of_steps = int((initial_range[1] - initial_range[0]) / step_size) + 1
    # construct the piecewise linear approximation of the silu function
    piecewise_function = combined_function_generator(silu, piecewise_function_segments, total_number_of_steps, False)
    # compute the mean and max error of the approximation compared to the silu function
    x = np.linspace(initial_range[0], initial_range[1], total_number_of_steps)
    silu_reference_val = silu(x)
    silu_approximation_val = piecewise_function(x)
    mean_error = np.mean(np.abs(silu_reference_val - silu_approximation_val))
    max_error = np.max(np.abs(silu_reference_val - silu_approximation_val))
    # construct the reward function
    # the reward function takes the linear weighted sum of mean and max error and takes a form of a
    # monotonic decreasing function whose value is between 0 and 10
    # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
    reward = input_reward_function(mean_error, max_error)

    if verbose:
        print("mean error: ", mean_error)
        print("max error: ", max_error)
        print("reward: ", reward)

        # visualize the function in matplotlib
        plt.plot(x, silu_reference_val, label="silu")
        plt.plot(x, silu_approximation_val, label="approximation")
        plt.scatter(np.array(chosen_points), piecewise_function(np.array(chosen_points)), color='red', label=f"chosen points: {len(chosen_points)}")
        plt.legend()
        plt.show()
        plt.close()


    return reward, mean_error, max_error

# takes in the list of all points chosen by the agent and the initial range
# then it constructs a piecewise linear approximation of the sigmoid function
# it computes the mean and max error of the approximation compared to the sigmoid function
# with the mean and max error it constructs a reward function that is used by the agent
def final_reward_function_sigmoid(chosen_points, initial_range, input_reward_function=common_reward_function):
    # append the min value of the range and the max value of the range to the first and last position of the list
    # with chosen_points filling the middle
    # this is needed as the piecewise linear function constructor needs the first and last point (the range) to be the min and max
    piecewise_function_segments = []
    piecewise_function_segments.append(initial_range[0])
    piecewise_function_segments.extend(chosen_points)
    if chosen_points[-1] != initial_range[1]:
        piecewise_function_segments.append(initial_range[1])
    step_size = 0.001 # determines the accuracy of the least square fit that generates the approximation of the sigmoid function
    total_number_of_steps = int((initial_range[1] - initial_range[0]) / step_size) + 1
    # construct the piecewise linear approximation of the sigmoid function
    piecewise_function = combined_function_generator(sigmoid, piecewise_function_segments, total_number_of_steps,
                                                     is_sigmoid=True)
    # compute the mean and max error of the approximation compared to the sigmoid function
    x = np.linspace(initial_range[0], initial_range[1], total_number_of_steps)
    sigmoid_reference_val = sigmoid(x)
    sigmoid_approximation_val = piecewise_function(x)
    mean_error = np.mean(np.abs(sigmoid_reference_val - sigmoid_approximation_val))
    max_error = np.max(np.abs(sigmoid_reference_val - sigmoid_approximation_val))
    # construct the reward function
    # the reward function takes the linear weighted sum of mean and max error and takes a form of a
    # monotonic decreasing function whose value is between 0 and 10
    # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
    reward = input_reward_function(mean_error, max_error)

    return reward

# takes in the list of all points chosen by the agent and the initial range
# then it constructs a piecewise linear approximation of the sigmoid function
# it computes the mean and max error of the approximation compared to the sigmoid function
# with the mean and max error it constructs a reward function that is used by the agent
# Returns the reward, mean error and max error
def final_reward_error_function_sigmoid(chosen_points, initial_range, verbose = False, input_reward_function=common_reward_function):
    # append the min value of the range and the max value of the range to the first and last position of the list
    # with chosen_points filling the middle
    # this is needed as the piecewise linear function constructor needs the first and last point (the range) to be the min and max
    piecewise_function_segments = []
    piecewise_function_segments.append(initial_range[0])
    piecewise_function_segments.extend(chosen_points)
    if chosen_points[-1] != initial_range[1]:
        piecewise_function_segments.append(initial_range[1])
    step_size = 0.001 # determines the accuracy of the least square fit that generates the approximation of the sigmoid function
    total_number_of_steps = int((initial_range[1] - initial_range[0]) / step_size) + 1
    # construct the piecewise linear approximation of the sigmoid function
    piecewise_function = combined_function_generator(sigmoid, piecewise_function_segments, total_number_of_steps,
                                                     is_sigmoid=True)
    # compute the mean and max error of the approximation compared to the sigmoid function
    x = np.linspace(initial_range[0], initial_range[1], total_number_of_steps)
    sigmoid_reference_val = sigmoid(x)
    sigmoid_approximation_val = piecewise_function(x)
    mean_error = np.mean(np.abs(sigmoid_reference_val - sigmoid_approximation_val))
    max_error = np.max(np.abs(sigmoid_reference_val - sigmoid_approximation_val))
    # construct the reward function
    # the reward function takes the linear weighted sum of mean and max error and takes a form of a
    # monotonic decreasing function whose value is between 0 and 10
    # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
    reward = input_reward_function(mean_error, max_error)
    if verbose:
        print("mean error: ", mean_error)
        print("max error: ", max_error)
        print("reward: ", reward)

        # visualize the function in matplotlib
        plt.plot(x, sigmoid_reference_val, label="sigmoid")
        plt.plot(x, sigmoid_approximation_val, label="approximation")
        plt.scatter(np.array(chosen_points), piecewise_function(np.array(chosen_points)), color='red', label=f"chosen points: {len(chosen_points)}")
        plt.legend()
        plt.show()
        plt.close()
    return reward, mean_error, max_error


# takes in the list of all points chosen by the agent and the initial range
# then it constructs a piecewise linear approximation of the gelu function
# it computes the mean and max error of the approximation compared to the gelu function
# with the mean and max error it constructs a reward function that is used by the agent
def final_reward_function_gelu(chosen_points, initial_range, input_reward_function=common_reward_function):
    # append the min value of the range and the max value of the range to the first and last position of the list
    # with chosen_points filling the middle
    # this is needed as the piecewise linear function constructor needs the first and last point (the range) to be the min and max
    piecewise_function_segments = []
    piecewise_function_segments.append(initial_range[0])
    piecewise_function_segments.extend(chosen_points)
    if chosen_points[-1] != initial_range[1]:
        piecewise_function_segments.append(initial_range[1])
    step_size = 0.001 # determines the accuracy of the least square fit that generates the approximation of the gelu function
    total_number_of_steps = int((initial_range[1] - initial_range[0]) / step_size) + 1
    # construct the piecewise linear approximation of the gelu function
    piecewise_function = combined_function_generator(gelu, piecewise_function_segments, total_number_of_steps,
                                                     is_sigmoid=False)
    # compute the mean and max error of the approximation compared to the gelu function
    x = np.linspace(initial_range[0], initial_range[1], total_number_of_steps)
    gelu_reference_val = gelu(x)
    gelu_approximation_val = piecewise_function(x)
    mean_error = np.mean(np.abs(gelu_reference_val - gelu_approximation_val))
    max_error = np.max(np.abs(gelu_reference_val - gelu_approximation_val))
    # construct the reward function
    # the reward function takes the linear weighted sum of mean and max error and takes a form of a
    # monotonic decreasing function whose value is between 0 and 10
    # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
    reward = input_reward_function(mean_error, max_error)

    return reward

# takes in the list of all points chosen by the agent and the initial range
# then it constructs a piecewise linear approximation of the gelu function
# it computes the mean and max error of the approximation compared to the gelu function
# with the mean and max error it constructs a reward function that is used by the agent
# Returns the reward, mean error and max error
def final_reward_error_function_gelu(chosen_points, initial_range, verbose = False, input_reward_function=common_reward_function):
    # append the min value of the range and the max value of the range to the first and last position of the list
    # with chosen_points filling the middle
    # this is needed as the piecewise linear function constructor needs the first and last point (the range) to be the min and max
    piecewise_function_segments = []
    piecewise_function_segments.append(initial_range[0])
    piecewise_function_segments.extend(chosen_points)
    if chosen_points[-1] != initial_range[1]:
        piecewise_function_segments.append(initial_range[1])
    step_size = 0.001 # determines the accuracy of the least square fit that generates the approximation of the gelu function
    total_number_of_steps = int((initial_range[1] - initial_range[0]) / step_size) + 1
    # construct the piecewise linear approximation of the gelu function
    piecewise_function = combined_function_generator(gelu, piecewise_function_segments, total_number_of_steps,
                                                     is_sigmoid=False)
    # compute the mean and max error of the approximation compared to the gelu function
    x = np.linspace(initial_range[0], initial_range[1], total_number_of_steps)
    gelu_reference_val = gelu(x)
    gelu_approximation_val = piecewise_function(x)
    mean_error = np.mean(np.abs(gelu_reference_val - gelu_approximation_val))
    max_error = np.max(np.abs(gelu_reference_val - gelu_approximation_val))
    # construct the reward function
    # the reward function takes the linear weighted sum of mean and max error and takes a form of a
    # monotonic decreasing function whose value is between 0 and 10
    # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
    reward = input_reward_function(mean_error, max_error)
    if verbose:
        print("mean error: ", mean_error)
        print("max error: ", max_error)
        print("reward: ", reward)

        # visualize the function in matplotlib
        plt.plot(x, gelu_reference_val, label="gelu")
        plt.plot(x, gelu_approximation_val, label="approximation")
        plt.scatter(np.array(chosen_points), piecewise_function(np.array(chosen_points)), color='red', label=f"chosen points: {len(chosen_points)}")
        plt.legend()
        plt.show()
        plt.close()
    return reward, mean_error, max_error



# Given a function and a list of points, this function will plot the function and the piecewise linear approximation based on the points
# And also mark the points on the plot
def plot_chosen_points(original_function, chosen_points, initial_range, function_name='silu', show_plot = False, save_fig_name = None):
    piecewise_function_segments = []
    piecewise_function_segments.append(initial_range[0])
    piecewise_function_segments.extend(chosen_points)
    if chosen_points[-1] != initial_range[1]:
        piecewise_function_segments.append(initial_range[1])
    step_size = 0.001 # determines the accuracy of the least square fit that generates the approximation of the silu function
    total_number_of_steps = int((initial_range[1] - initial_range[0]) / step_size) + 1
    # construct the piecewise linear approximation of the silu function
    piecewise_function = combined_function_generator(original_function, piecewise_function_segments, total_number_of_steps, False)
    # compute the mean and max error of the approximation compared to the silu function
    x = np.linspace(initial_range[0], initial_range[1], total_number_of_steps)
    silu_reference_val = original_function(x)
    silu_approximation_val = piecewise_function(x)
    # visualize the function in matplotlib
    # visualize the function in matplotlib
    plt.plot(x, silu_reference_val, label=function_name)
    plt.plot(x, silu_approximation_val, label="approximation")
    plt.scatter(np.array(chosen_points), piecewise_function(np.array(chosen_points)), color='red',
                label=f"chosen points: {len(chosen_points)}")
    plt.legend()
    if show_plot:
        plt.show()
    if save_fig_name is not None:
        plt.savefig(save_fig_name)
    plt.close()

# Implementation taken from segment_random_search_baseline.py
# Expends upon the functionality of plot_chosen_points by also plotting the x coordinates of the chosen points
def plot_chosen_points_x_coord_draft(original_function, chosen_points, initial_range, reward, mean_error, max_error,
                                     function_name='silu', show_plot=False, save_fig_name=None):
    piecewise_function_segments = []
    piecewise_function_segments.append(initial_range[0])
    piecewise_function_segments.extend(chosen_points)
    if chosen_points[-1] != initial_range[1]:
        piecewise_function_segments.append(initial_range[1])
    step_size = 0.001  # determines the accuracy of the least square fit that generates the approximation of the silu function
    total_number_of_steps = int((initial_range[1] - initial_range[0]) / step_size) + 1
    # construct the piecewise linear approximation of the silu function
    piecewise_function = combined_function_generator(original_function, piecewise_function_segments,
                                                     total_number_of_steps, False)
    # compute the mean and max error of the approximation compared to the silu function
    x = np.linspace(initial_range[0], initial_range[1], total_number_of_steps)
    reference_val = original_function(x)
    approximation_val = piecewise_function(x)
    # visualize the function in matplotlib
    if show_plot:
        # visualize the function in matplotlib
        plt.figure(figsize=(10, 6), dpi=100)  # Set the figure size
        plt.plot(x, reference_val, label=function_name)
        plt.plot(x, approximation_val, label="approximation")
        plt.scatter(np.array(chosen_points), piecewise_function(np.array(chosen_points)), color='red',
                    label=f"chosen points: {len(chosen_points)}")
        # Add x-coordinate text near the points
        for i, point in enumerate(chosen_points):
            plt.annotate(f"{point:.3f}", (point, piecewise_function(point)), xytext=(5, -10),
                         textcoords='offset points', ha='left', va='top')
        # Add reward, mean error, and max error to the legends as text and numbers
        legend_text = f"reward: {reward:.3f}, mean error: {mean_error:.3f}, max error: {max_error:.3f}"
        plt.legend(title=legend_text)
        plt.show()
        plt.close()
    if save_fig_name is not None:
        # Introduced a pause to allow the plot to be saved
        plt.pause(0.1)
        # visualize the function in matplotlib
        plt.figure(figsize=(10, 6), dpi=100)  # Set the figure size
        plt.plot(x, reference_val, label=function_name)
        plt.plot(x, approximation_val, label="approximation")
        plt.scatter(np.array(chosen_points), piecewise_function(np.array(chosen_points)), color='red',
                    label=f"chosen points: {len(chosen_points)}")
        # Add x-coordinate text near the points
        for i, point in enumerate(chosen_points):
            plt.annotate(f"{point:.3f}", (point, piecewise_function(point)), xytext=(5, -10),
                         textcoords='offset points', ha='left', va='top')
        # Add reward, mean error, and max error to the legends as text and numbers
        legend_text = f"reward: {reward:.3f}, mean error: {mean_error:.3f}, max error: {max_error:.3f}"
        plt.legend(title=legend_text)
        plt.savefig(save_fig_name)
        plt.close()



# One Cycle Learning Rate Schedule
# There is a warmup period for 0.1 of the total number of steps
# Where the learning rate increases linearly from 0 to the initial learning rate
# Then there is a linear decay from the initial learning rate to 0 for the rest of the steps
def OneCycleLR_Schedule_Linear_MinLR(initial_value: float, warmup_proportion: float = 0.1, multiplier: float = 20.0) -> \
Callable[[float], float]:
    """
    OneCycle learning rate schedule with linear warmup and linear decay.
    :param initial_value: Initial learning rate.
    :param warmup_proportion: Proportion of total steps allocated to warmup.
    :param multiplier: The factor used to compute the minimum end learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    assert 0.0 < warmup_proportion < 1.0, "Warmup proportion must be between 0 and 1."
    min_lr = initial_value / multiplier

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > (1 - warmup_proportion):
            # Warmup phase: the learning rate increases linearly from 0 to the initial learning rate.
            warmup_progress = (progress_remaining - 1) * (-1 / warmup_proportion)
            return initial_value * warmup_progress
        else:
            # Linear decay phase: the learning rate decreases linearly from the initial learning rate to min_lr.
            decay_progress = progress_remaining / (1 - warmup_proportion)  # goes from 1 -> 0
            return (initial_value - min_lr) * decay_progress + min_lr

    return func


# One Cycle Learning Rate Schedule Cosine Annealing
def OneCycleLR_Schedule_Cosine(initial_value: float, warmup_proportion: float = 0.1, multiplier: float = 20.0) -> \
Callable[[float], float]:
    """
    OneCycle learning rate schedule with linear warmup and cosine annealing.
    :param initial_value: Initial learning rate.
    :param warmup_proportion: Proportion of total steps allocated to warmup.
    :param multiplier: The factor used to compute the minimum end learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    assert 0.0 < warmup_proportion < 1.0, "Warmup proportion must be between 0 and 1."
    min_lr = initial_value / multiplier

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > (1 - warmup_proportion):
            # Warmup phase: the learning rate increases linearly from 0 to the initial learning rate.
            warmup_progress = (progress_remaining - 1) * (-1 / warmup_proportion)
            return initial_value * warmup_progress
        else:
            # Cosine annealing phase: the learning rate decreases from the initial learning rate to min_lr following a half cosine curve.
            annealing_progress = (progress_remaining / (warmup_proportion - 1)) + 1  # goes from 0 -> 1
            return (initial_value - min_lr) * 0.5 * (1 + np.cos(np.pi * annealing_progress)) + min_lr

    return func



# One Cycle Learning Rate Schedule Exponential Annealing
def OneCycleLR_Schedule_Exponential(initial_value: float, warmup_proportion: float = 0.1, multiplier: float = 20.0) -> \
Callable[[float], float]:
    """
    OneCycle learning rate schedule with exponential warmup and exponential decay.
    :param initial_value: Initial learning rate.
    :param warmup_proportion: Proportion of total steps allocated to warmup.
    :param multiplier: The factor used to compute the minimum end learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    assert 0.0 < warmup_proportion < 1.0, "Warmup proportion must be between 0 and 1."
    min_lr = initial_value / multiplier

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > (1 - warmup_proportion):
            # Warmup phase: the learning rate increases linearly from 0 to the initial learning rate.
            warmup_progress = (progress_remaining - 1) * (-1 / warmup_proportion)
            return initial_value * warmup_progress
        else:
            # Exponential decay phase: the learning rate decreases exponentially from the initial learning rate to min_lr.
            decay_progress = (progress_remaining / (warmup_proportion - 1)) + 1  # goes from 0 -> 1
            return (initial_value - min_lr) * np.exp(-5 * decay_progress) + min_lr

    return func

# Define the RL environment using gym
# function with range (-8, 8) and 10 points, which can be changed during initialization
# Each round an agent will choose a number inside the initial range.
# Then the agent will receive a reward of silu_curvature_alt(chosen_number) and the range will be updated.
# The agent will receive a reward of 0 if it chooses a number outside the range and the entire game will be reset.
# If the agent chooses a number inside the range, the new range will be (chosen_number, old_range[1])
# After all the points are used, the game will be reset.
# If the game ends without violating the range condition midway there will be a final reward of final_reward_function_silu(chosen_points, initial_range)
# The agent will receive a reward of 0 if it chooses a number outside the range and the entire game will be reset.
class RL_Environment(gym.Env):
    def __init__(self, initial_range=(-8, 8), num_points=10, action_increment=0.01):
        super(RL_Environment, self).__init__()
        self.initial_range = initial_range
        self.range = self.initial_range
        self.total_points = num_points
        self.points_left = self.total_points
        self.chosen_points = []

        # Change action space to be Discrete
        self.num_actions = int((self.initial_range[1] - self.initial_range[0]) / action_increment) + 1
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # Define the observation space as a Box space
        # observation space (number_of_remaining_points, last_chosen_point, space_between_end_of_range_and_chosen_point)
        low_obs = np.array([0, self.initial_range[0], 0.])
        high_obs = np.array([self.total_points, self.initial_range[1], self.initial_range[1] - self.initial_range[0]])
        self.observation_space = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # # For conversion from discrete action to continuous value
        # self.action_values = np.linspace(self.initial_range[0], self.initial_range[1], self.num_actions)

    def reset(self, seed=None, **kwargs):
        self.range = self.initial_range
        self.points_left = self.total_points
        self.chosen_points = []
        # return value from reset should be: observation, info
        return np.array([self.points_left, self.range[0], self.initial_range[1] - self.initial_range[0]]), {}

    def step(self, action):
        # Execute one step within the environment
        chosen_point = action * (self.initial_range[1] - self.initial_range[0]) / (self.num_actions - 1) + self.range[0]
        # Check if the chosen point is in the range
        if self.range[0] <= chosen_point <= self.range[1]:
            self.chosen_points.append(chosen_point)
            self.range = (chosen_point, self.range[1])
            self.points_left -= 1
        else:
            # If not, return a reward of -10 and reset, heavily penalize early truncation
            reward = -10

            # when agent goes out of bounds
            chosen_point = self.initial_range[1]
            self.chosen_points.append(chosen_point)
            self.range = (chosen_point, self.range[1])

            # return value from step should be: observation, reward, terminated, truncated, info
            return np.array(
                [self.points_left, self.range[0], self.initial_range[1] - self.range[0]]), reward, False, True, {}

        # Check if all points are used up
        if self.points_left == 0:
            # Calculate final reward and reward for the chosen point
            final_reward = final_reward_function_silu(self.chosen_points, self.initial_range)
            reward_for_choosing_point = silu_curvature_alt(chosen_point)
            combined_reward = final_reward + reward_for_choosing_point
            # return value from step should be: observation, reward, terminated, truncated, info
            return np.array([self.points_left, self.range[0], self.initial_range[1] - self.range[0]]), combined_reward, True, False, {}
        else:
            # Calculate reward for the chosen point
            reward = silu_curvature_alt(chosen_point)
            # return value from step should be: observation, reward, terminated, truncated, info
            return np.array([self.points_left, self.range[0], self.initial_range[1] - self.range[0]]), reward, False, False, {}

# Implementation copied and modified from RL_Environment.py (original RL_Environment_test2_continuous_action_space)
# Changes the original RL_Environment to have an observation space separated into 3 parts using Dict
# This allowed the remaining points to be a Discrete space, while the last chosen point and the space between the end of the range and the chosen point to be a Box space
# This allows more accurate representation
# The action space was changed to be continuous, it used to have something like 1600 actions, which is too much for the agent to learn
# Each action used to represent a point in the range and did not represent movement
# now each action represents a point in the range normalized to [-1, 1]=>(0, 0.5 * (initial_range[1] - initial_range[0]))
# RL_Environment_test2_continuous_action_space is changed to take in functions for the reward
# It is able to accommodate other functions besides SiLU
class RL_Environment_test2_continuous_action_space_generalized(gym.Env):
    def __init__(self, initial_range, num_points, input_curvature_function, input_final_reward_function):
        super(RL_Environment_test2_continuous_action_space_generalized, self).__init__()
        self.initial_range = initial_range
        self.range = self.initial_range
        self.total_points = num_points
        self.points_left = self.total_points
        self.chosen_points = []
        self.curvature_function = input_curvature_function
        self.final_reward_function = input_final_reward_function

        # Change action space to be continuous
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)


        # Define the observation space as a Dict space
        num_points_space = gym.spaces.Discrete(self.total_points + 1)
        last_point_space = gym.spaces.Box(low=np.array([self.initial_range[0]], dtype=np.float32),
                                          high=np.array([self.initial_range[1]], dtype=np.float32))
        space_to_end_space = gym.spaces.Box(low=np.array([0.], dtype=np.float32),
                                            high=np.array([self.initial_range[1] - self.initial_range[0]], dtype=np.float32))

        self.observation_space = gym.spaces.Dict({
            'num_points_left': num_points_space,
            'last_chosen_point': last_point_space,
            'space_to_end': space_to_end_space
        })

    def reset(self, seed=None, **kwargs):
        self.range = self.initial_range
        self.points_left = self.total_points
        self.chosen_points = []
        # return value from reset should be: observation, info
        return {
            'num_points_left': self.points_left,
            'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
            'space_to_end': np.array([self.initial_range[1] - self.initial_range[0]], dtype=np.float32)
        }, {}

    def step(self, action):
        # Execute one step within the environment
        initial_range_length = self.initial_range[1] - self.initial_range[0]
        last_chosen_point = self.range[0] if self.chosen_points else self.initial_range[0]
        # action is a number between -1 and 1, or [-1, 1]
        # this means that the agent takes a step of [0, 1/2 of the initial range] starting from last_chosen_point
        # if there is no last_chosen_point, then the agent starts from the left most point of initial range
        chosen_point = (action[0] + 1) * (initial_range_length / 4) + last_chosen_point

        # Check if the chosen point is in the range
        if self.range[0] <= chosen_point <= self.range[1]:
            self.chosen_points.append(chosen_point)
            self.range = (chosen_point, self.range[1])
            self.points_left -= 1
        else:
            # If not, return a reward of -100 and reset, heavily penalize early truncation
            reward = -100
            # when agent goes out of bounds
            chosen_point = self.initial_range[1]
            self.chosen_points.append(chosen_point)
            self.range = (chosen_point, self.range[1])
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, reward, False, True, {}

        # Check if all points are used up
        if self.points_left == 0:
            # Calculate final reward and reward for the chosen point
            final_reward = self.final_reward_function(self.chosen_points, self.initial_range)  # this function should be defined
            reward_for_choosing_point = self.curvature_function(chosen_point)  # this function should be defined
            combined_reward = final_reward + reward_for_choosing_point
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, combined_reward, True, False, {}
        else:
            # Calculate reward for the chosen point
            reward = self.curvature_function(chosen_point)  # this function should be defined
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, reward, False, False, {}

# Functions used for generating nonlinear mapping
# P is the hyperparameter that controls the range of the mapping and represents 0.5 of initial range length
# It takes general logistic function of the form P / (C + Q * exp(B * x))
# and finds the parameters C, Q, B that satisfy the following equations: (P being the input hyperparameter)
# P / (C + Q * exp(B)) = 0.005 * P
# P / (C + Q) = 1
# P / (C + Q * exp(-B)) = 0.99 * P
# The solution is found using fsolve from scipy.optimize
# The function asymmetric_map is the nonlinear mapping function
# It is able to map a number between -1 and 1 to a number between 0.005 * P and 0.99 * P
# Able to focus on the 0~1 of the output where most of the action takes place
def configure_equations(P):
    def return_equation(x):
        C, Q, B = x

        eq1 = P / (C + Q * np.exp(B)) - 0.01 * P
        eq2 = P / (C + Q) - 2
        eq3 = P / (C + Q * np.exp(-B)) - 0.99 * P

        return [eq1, eq2, eq3]
    return return_equation

def asymmetric_map(x, C, Q, B, P):
    return P / (C + Q * np.exp(-B * x))

def form_asymmetric_map_function(P, initial_guess=(1, 1, 1)):
    P_equations = configure_equations(P)
    solution = fsolve(P_equations, initial_guess)
    C, Q, B = solution[0], solution[1], solution[2]
    return lambda x: asymmetric_map(x, C, Q, B, P)


# Iterate upon the previous environment, but this time, the curvature function is normalized, max is 0.2, min is 0
# Also the action[-1,1] is mapped nonlinearly, previously the mapping was linear but now the mapping uses
# a generalized logistic function
# Previously linear mapping [-1, 1] => [0, 0.5 * initial_range_length]
# Now generalized logistic mapping: -1 => 0.005 * initial_range_length, 0 1=> 1, 1 => 0.99 * initial_range_length
# This was done because most spaces between points are very small, so the agent should be able to take smaller steps
# The re-mapping prioritizes smaller steps which are more common
class RL_Environment_test2_continuous_action_space_generalized_nonlinear_map_normalized_curvature(gym.Env):
    def __init__(self, initial_range, num_points, input_curvature_function, input_reward_function_definition, input_final_reward_function):
        super(RL_Environment_test2_continuous_action_space_generalized_nonlinear_map_normalized_curvature, self).__init__()
        self.initial_range = initial_range
        self.range = self.initial_range
        self.total_points = num_points
        self.points_left = self.total_points
        self.chosen_points = []
        self.curvature_function = input_curvature_function
        self.reward_function_definition = input_reward_function_definition
        self.final_reward_function = input_final_reward_function
        # Curvature normalization parameter
        self.max_curvature_reward = 0.5
        # Takes maximum curvature reward and divides it by the maximum curvature value within range
        self.curvature_normalization_parameter = \
            self.max_curvature_reward / np.max(np.arange(self.initial_range[0], self.initial_range[1], 0.01))

        # Action space nonlinear mapping function
        # Maps normalized action between [-1, 1] into actual point coordinates
        self.action_space_mapping_function = \
            form_asymmetric_map_function(0.5 * (self.initial_range[1] - self.initial_range[0]))

        # Change action space to be continuous
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)


        # Define the observation space as a Dict space
        num_points_space = gym.spaces.Discrete(self.total_points + 1)
        last_point_space = gym.spaces.Box(low=np.array([self.initial_range[0]], dtype=np.float32),
                                          high=np.array([self.initial_range[1]], dtype=np.float32))
        space_to_end_space = gym.spaces.Box(low=np.array([0.], dtype=np.float32),
                                            high=np.array([self.initial_range[1] - self.initial_range[0]], dtype=np.float32))

        self.observation_space = gym.spaces.Dict({
            'num_points_left': num_points_space,
            'last_chosen_point': last_point_space,
            'space_to_end': space_to_end_space
        })

    def reset(self, seed=None, **kwargs):
        self.range = self.initial_range
        self.points_left = self.total_points
        self.chosen_points = []
        # return value from reset should be: observation, info
        return {
            'num_points_left': self.points_left,
            'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
            'space_to_end': np.array([self.initial_range[1] - self.initial_range[0]], dtype=np.float32)
        }, {}

    def step(self, action):
        # Execute one step within the environment
        initial_range_length = self.initial_range[1] - self.initial_range[0]
        last_chosen_point = self.range[0] if self.chosen_points else self.initial_range[0]
        if self.chosen_points == []:
            last_chosen_point = self.initial_range[0]
        else:
            last_chosen_point = self.chosen_points[-1]
        # action is a number between -1 and 1, or [-1, 1]
        # this means that the agent takes a step of [0.005 of the 1/2 of the initial range, 0.99 of the 1/2 of the initial range] starting from last_chosen_point
        # if there is no last_chosen_point, then the agent starts from the left most point of initial range
        ''' 
        chosen_point = self.action_space_mapping_function(action[0]) + last_chosen_point
        '''

        # Previous linear mapping function, now replaced with generalized logistic function
        # due to poor result from nonlinear mapping(plateaued too early), revert to linear mapping
        # except adding a small constant to the chosen_point to prevent the agent from choosing the same point again
        chosen_point = (action[0] + 1) * (initial_range_length / 4) + last_chosen_point + 0.1


        # Check if the chosen point is in the range, use < on left side to avoid duplicate points
        if self.range[0] < chosen_point <= self.range[1]:
            # give penalty if the points are too close to each other
            # check previous two points in chosen_points and give penalty if furthest distance is within 0.2
            clustering_penalty = 0
            if len(self.chosen_points) >= 2:
                if chosen_point - self.chosen_points[-2] < 0.21:
                    clustering_penalty = clustering_penalty - 5
            elif len(self.chosen_points) == 1:
                if chosen_point - self.chosen_points[-1] < 0.11:
                    clustering_penalty = clustering_penalty - 5
            else:
                if chosen_point - self.initial_range[0] < 0.11:
                    clustering_penalty = clustering_penalty - 5

            self.chosen_points.append(chosen_point)
            self.range = (chosen_point, self.range[1])
            self.points_left -= 1
        else:
            # If not, return a reward of -10 and reset, heavily penalize early truncation
            reward = -10
            # when agent goes out of bounds
            chosen_point = self.initial_range[1]
            self.chosen_points.append(chosen_point)
            self.range = (chosen_point, self.range[1])
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, reward, False, True, {}

        # Check if all points are used up
        if self.points_left == 0:
            # Calculate final reward and reward for the chosen point
            final_reward = self.final_reward_function(self.chosen_points, self.initial_range, self.reward_function_definition)  # this function should be defined
            reward_for_choosing_point = self.curvature_normalization_parameter * self.curvature_function(chosen_point)  # this function should be defined
            # check if there are at least 2 points left of middle of range, give small reward if true
            even_distribution_points = 0
            left_points = 0
            for point in self.chosen_points:
                if point < (self.initial_range[0] + self.initial_range[1]) / 2:
                    left_points += 1
            if left_points >= 2:
                even_distribution_points = even_distribution_points + 1
            combined_reward = final_reward + reward_for_choosing_point + clustering_penalty + even_distribution_points
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, combined_reward, True, False, {}
        else:
            # Calculate reward for the chosen point
            reward = self.curvature_normalization_parameter * self.curvature_function(chosen_point)  # this function should be defined
            combined_reward = reward + clustering_penalty
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, combined_reward, False, False, {}


# Training function
def train_ppo(test_enabled=True, initial_range=(-8, 8), num_points=10, learning_rate=0.0002, train_timesteps= 10_000, save_model=False, save_logs=False, verbose=True):
    # Create a vectorized environment with custom range and number of points

    env = DummyVecEnv([lambda: RL_Environment(initial_range=initial_range, num_points=num_points)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # Initialize the PPO model
    if save_logs:
        log_dir = "./SiLU_approx_tensorboard_logs/"
    else:
        log_dir = None
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=int(verbose), tensorboard_log=log_dir)

    # Train the model
    model.learn(total_timesteps=train_timesteps)

    # get timestamp to name the model with number of points and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    # Save the trained model
    if save_model:
        model.save(os.path.join('model_archive', f"ppo_silu_approx_{num_points}_points_{timestamp}.zip"))

    reward = None
    mean_error = None
    max_error = None
    final_chosen_points = None

    # Test the model
    if test_enabled:
        # repeat the process for iter_num and then pick best one

        n_steps = num_points

        # Create a vectorized environment with custom range and number of points
        env = DummyVecEnv([lambda: RL_Environment(initial_range=initial_range, num_points=num_points)])
        obs = env.reset()


        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            if verbose:
                print("Step {}".format(step + 1))
                print("Action: ", action)
            l = env.envs[0].chosen_points
            obs, reward, done, info = env.step(action)
            if verbose:
                print('Observation: ', obs)
                print('Reward: ', reward)
                print('Done: ', done)
                print('Chosen Points: ', env.envs[0].chosen_points)
                print('Remaining Points', env.envs[0].points_left)
            if done:
                if verbose:
                    print("Goal reached!", "reward=", reward)
                # new_l = np.append(l, action * (env.envs[0].initial_range[1] - env.envs[0].initial_range[0]) / (env.envs[0].num_actions - 1) + max(env.envs[0].initial_range[0], l[-1]))
                # l.append(action * (env.envs[0].initial_range[1] - env.envs[0].initial_range[0]) / (env.envs[0].num_actions - 1) + env.envs[0].range[0])
                reward, mean_error, max_error = final_reward_function_silu_print(l, env.envs[0].initial_range, verbose=verbose)
                final_chosen_points = l
                if verbose:
                    print('Final Chosen Points: ', final_chosen_points)
                # Final Chosen Points:  [-3.59, -2.5999999999999996, -1.6099999999999997, -1.4699999999999998, -0.47999999999999976, -0.15999999999999975, -0.009999999999999759, 0.3100000000000003, 0.4600000000000003, 0.7800000000000002]
                break

    return model, final_chosen_points, reward, mean_error, max_error


def plot_overlapping_histogram(A, B, C, figure_legend=None, figure_name=None):
    # Close any previously open figures if it exists
    plt.close()

    # Set figure size and margins
    plt.figure(figsize=(10, 6))
    plt.margins(0.02)

    # Set histogram parameters
    bins = 20
    alpha = 0.5

    # Plot histograms for each dataset
    plt.hist(A, bins=bins, density=True, alpha=alpha, color='blue', label=figure_legend[0] if figure_legend else None)
    plt.hist(B, bins=bins, density=True, alpha=alpha, color='orange', label=figure_legend[1] if figure_legend else None)
    plt.hist(C, bins=bins, density=True, alpha=alpha, color='green', label=figure_legend[2] if figure_legend else None)

    # Add labels and a legend
    plt.xlabel('X-axis')
    plt.ylabel('Density')
    plt.legend()


    if figure_name is not None:
        # Save the figure
        plt.savefig(figure_name)

    # Display the plot
    plt.show()
    plt.close()


# Takes in various arguments like function name, range, number of points, etc. and returns result and model
# Completely configurable in terms of different functions and number of points
# Uses the new modified environment with Dict space and continuous action space
# Derived for use in RL_Environment_test2_continuous_action_space
def single_train_run_function(
        input_function_name,
        input_num_points,
        input_initial_range,
        input_function,
        input_curvature_function,
        input_reward_function_definition,
        input_final_reward_function,
        input_final_reward_error_function,
        input_train_timesteps,
        input_environment,
        input_verbose,
        input_algorithm_name,
        input_algorithm,
        input_policy,
        input_learning_rate, # Can be a constant number like 0.001 or Schedule like OneCycleLR_Schedule_Linear_MinLR
        learning_rate_configs, # dictionary that contains initial_value, warmup_proportion, and multiplier for schedule, if learning rate is constant this is ignored
        algorithm_parameters
):
    # Get the time stamp for the run
    run_time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a directory for the run
    run_dir = os.path.join('run_archive', f'{input_function_name}_{input_num_points}_points_{run_time_stamp}_single_model_run_train_timestep_{input_train_timesteps}')
    os.mkdir(run_dir)
    # Set up environment for run
    monitor_env = Monitor(env=input_environment(initial_range=input_initial_range, num_points=input_num_points,
                                                input_curvature_function=input_curvature_function,
                                                input_reward_function_definition=input_reward_function_definition,
                                                input_final_reward_function=input_final_reward_function
                                                 ),
                      filename=os.path.join(run_dir, f'run_monitor_{run_time_stamp}.csv'),
                      )
    run_env = DummyVecEnv([lambda: monitor_env])

    tensorboard_name = f'{input_algorithm_name}_{input_function_name}_{input_num_points}_points_train_timestep_{input_train_timesteps}_single_model_run_{run_time_stamp}'

    # When the input_learning_rate is a constant
    if type(input_learning_rate) == float or type(input_learning_rate) == int:
        algorithm_learning_rate_input = input_learning_rate
    else: # When the input_learning_rate is a schedule
        algorithm_learning_rate_input = input_learning_rate(**learning_rate_configs)


    # Create the model
    model = input_algorithm(policy=input_policy, env=run_env, learning_rate=algorithm_learning_rate_input,
                            verbose=input_verbose, tensorboard_log='test_environment_tensorboard_log',
                            **algorithm_parameters)

    # Initialize the callback
    eval_env = DummyVecEnv([lambda: input_environment(initial_range=input_initial_range, num_points=input_num_points,
                                                      input_curvature_function=input_curvature_function,
                                                      input_reward_function_definition=input_reward_function_definition,
                                                      input_final_reward_function=input_final_reward_function
                                                      )])
    callback = EvalCallback(eval_env, best_model_save_path=run_dir,
                            log_path=run_dir, eval_freq=2048, # make eval_freq be same as batch size
                            deterministic=True, render=False,
                            verbose=input_verbose)
    model.learn(total_timesteps=input_train_timesteps,
                tb_log_name=tensorboard_name,
                progress_bar=True,
                callback=callback)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, run_env, n_eval_episodes=10)
    print('Final Model Evaluation Results:')
    print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')

    # Save the model
    model_name = f'{input_algorithm_name}_{input_function_name}_{input_num_points}_points_train_timestep_{input_train_timesteps}_single_model_run_{run_time_stamp}'
    model.save(os.path.join(run_dir, f'{model_name}_final_model'))

    # Get the final chosen points
    obs = run_env.reset()
    for step in range(input_num_points):
        action, _ = model.predict(obs, deterministic=True)
        l = run_env.envs[0].chosen_points
        obs, mid_run_reward, done, info = run_env.step(action)
        if done:
            final_reward, final_mean_error, final_max_error = input_final_reward_error_function(l, input_initial_range, verbose=False, input_reward_function=input_reward_function_definition)
            final_chosen_points = l
            print('Final Chosen Points: ', final_chosen_points)
            print('Final Reward: ', final_reward)
            print('Mean Error: ', final_mean_error)
            print('Max Error: ', final_max_error)
            # Plot the final chosen points and graph
            final_plot_name = os.path.join(run_dir, f'{model_name}_final_model_final_chosen_points_figure.png')
            plot_chosen_points_x_coord_draft(
                original_function=input_function,
                chosen_points=final_chosen_points,
                initial_range=input_initial_range,
                reward=final_reward,
                mean_error=final_mean_error,
                max_error=final_max_error,
                function_name=input_function_name,
                show_plot=True,
                save_fig_name=final_plot_name)
            # Save the logs and the final chosen points
            with open(os.path.join(run_dir, f'{model_name}_final_model_log.txt'), 'w') as f:
                f.write('Function: ' + input_function_name + '\n')
                f.write('Number of Points: ' + str(input_num_points) + '\n')
                f.write('Train Timesteps: ' + str(input_train_timesteps) + '\n')
                f.write('Policy: ' + input_algorithm_name + '\n')
                if type(input_learning_rate) == float or type(input_learning_rate) == int:
                    f.write('Learning Rate: ' + str(input_learning_rate) + '\n')
                else:
                    f.write('Learning Rate Schedule: ' + str(input_learning_rate.__name__) + '\n')
                    f.write('Learning Rate Schedule Configs: ' + str(learning_rate_configs) + '\n')
                f.write(f'Model Evaluation Results:\nMean Reward: {mean_reward}, Std Reward: {std_reward}\n')
                f.write(f'Final Chosen Points: {final_chosen_points}\n')
                f.write(f'Final Reward: {final_reward}\n')
                f.write(f'Mean Error: {final_mean_error}\n')
                f.write(f'Max Error: {final_max_error}\n')
                f.write(f'Run Time Stamp: {run_time_stamp}\n')
                f.write(f'Algorithm Parameters: {algorithm_parameters}\n')
            break

    # Get the evaluation result for best model
    best_model = input_algorithm.load(os.path.join(run_dir, 'best_model.zip'))
    # Evaluate the best model
    mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=10)
    print('Best Model Evaluation Results:')
    print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')

    # Get the final chosen points from best model
    obs = eval_env.reset()
    for step in range(input_num_points):
        action, _ = best_model.predict(obs, deterministic=True)
        l = eval_env.envs[0].chosen_points
        obs, mid_run_reward, done, info = eval_env.step(action)
        if done:
            final_reward, final_mean_error, final_max_error = input_final_reward_error_function(l, input_initial_range, verbose=False, input_reward_function=input_reward_function_definition)
            final_chosen_points = l
            print('Final Chosen Points: ', final_chosen_points)
            print('Final Reward: ', final_reward)
            print('Mean Error: ', final_mean_error)
            print('Max Error: ', final_max_error)
            # Plot the final chosen points and graph
            final_plot_name = os.path.join(run_dir, f'{model_name}_best_model_final_chosen_points_figure.png')
            plot_chosen_points_x_coord_draft(
                original_function=input_function,
                chosen_points=final_chosen_points,
                initial_range=input_initial_range,
                reward=final_reward,
                mean_error=final_mean_error,
                max_error=final_max_error,
                function_name=input_function_name,
                show_plot=True,
                save_fig_name=final_plot_name)
            # Save the logs and the final chosen points
            with open(os.path.join(run_dir, f'{model_name}_best_model_log.txt'), 'w') as f:
                f.write('Function: ' + input_function_name + '\n')
                f.write('Number of Points: ' + str(input_num_points) + '\n')
                f.write('Train Timesteps: ' + str(input_train_timesteps) + '\n')
                f.write('Policy: ' + input_algorithm_name + '\n')
                if type(input_learning_rate) == float or type(input_learning_rate) == int:
                    f.write('Learning Rate: ' + str(input_learning_rate) + '\n')
                else:
                    f.write('Learning Rate Schedule: ' + str(input_learning_rate.__name__) + '\n')
                    f.write('Learning Rate Schedule Configs: ' + str(learning_rate_configs) + '\n')
                f.write(f'Model Evaluation Results:\nMean Reward: {mean_reward}, Std Reward: {std_reward}\n')
                f.write(f'Final Chosen Points: {final_chosen_points}\n')
                f.write(f'Final Reward: {final_reward}\n')
                f.write(f'Mean Error: {final_mean_error}\n')
                f.write(f'Max Error: {final_max_error}\n')
                f.write(f'Run Time Stamp: {run_time_stamp}\n')
                f.write(f'Algorithm Parameters: {algorithm_parameters}\n')
            return final_reward, final_mean_error, final_max_error
# This is a function that was directly taken from the main portion of the RL_PPO_SB3.py code.
# It runs multiple training iterations and saves the best model by comparing its reward.
# This was due to the fact that before the environment was reworked
# in RL_Environment_Test and RL_Environment_test2_continuous_action_space
# the training was very unstable and required multiple iterations.
# This function is not currently used as the modifications in the requirement made multiple iterations unnecessary.
# It is kept here for reference.
# If multiple iterations are needed in the future, this function can be modified to work with the new environment.
def multi_train_run_function_DRAFT_WIP():
    initial_range = (-8, 8)
    max_num_points = 10
    # Start the training
    model, final_chosen_points, reward, mean_error, max_error = train_ppo(initial_range=initial_range, num_points=max_num_points)

    # short delay to prevent the tqdm progress bar from being printed before the earlier process outputs
    time.sleep(0.1)
    # after iter_num of training runs, pick the best one and save it
    iter_num = 10
    best_reward = float('-inf')
    best_model = None
    best_chosen_points = None
    best_mean_error = None
    best_max_error = None
    list_of_rewards = []
    list_of_mean_errors = []
    list_of_max_errors = []
    list_of_chosen_points = []
    list_of_models = []

    # add tqdm tracking over iterations

    for i in tqdm.tqdm(range(iter_num)):
        model, final_chosen_points, reward, mean_error, max_error = train_ppo(initial_range=initial_range, num_points=max_num_points, test_enabled=True, verbose=False)
        list_of_rewards.append(reward)
        list_of_mean_errors.append(mean_error)
        list_of_max_errors.append(max_error)
        list_of_chosen_points.append(final_chosen_points)
        list_of_models.append(model)
        if reward > best_reward:
            best_reward = reward
            best_model = model
            best_chosen_points = final_chosen_points
            best_mean_error = mean_error
            best_max_error = max_error
    # If final point equals end of range eliminate it
    if best_chosen_points is not None and best_chosen_points[-1] == initial_range[1]:
         best_chosen_points = best_chosen_points[:-1]

    # Output best results and save the model
    best_model_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print('Best Reward: ', best_reward)
    print('Best Chosen Points: ', best_chosen_points)
    print('Best Mean Error: ', best_mean_error)
    print('Best Max Error: ', best_max_error)
    final_reward_function_silu_print(best_chosen_points, initial_range, verbose=True)

    best_model.save(os.path.join('model_archive', f"ppo_silu_approx_best_model_in_run_{len(best_chosen_points)}_points_{best_model_timestamp}.zip"))

    # Also save the top percentile % of models in terms of reward value
    percentile = 5
    threshold = np.percentile(list_of_rewards, 100 - percentile)
    top_rewards = [reward for reward in list_of_rewards if reward >= threshold]
    top_models = [model for model, reward in zip(list_of_models, list_of_rewards) if reward >= threshold]
    top_mean_errors = [mean_error for mean_error, reward in zip(list_of_mean_errors, list_of_rewards) if reward >= threshold]
    top_max_errors = [max_error for max_error, reward in zip(list_of_max_errors, list_of_rewards) if reward >= threshold]
    top_chosen_points = [chosen_points for chosen_points, reward in zip(list_of_chosen_points, list_of_rewards) if reward >= threshold]
    run_dir = os.path.join('run_archive', f'silu_{max_num_points}_points_{best_model_timestamp}_top_{percentile}_percentile')
    os.mkdir(run_dir)
    # Save all the top percentile models
    for model_num in range(len(top_models)):
        current_model = top_models[model_num]
        model_save_path = os.path.join(run_dir, f"ppo_silu_approx_top_{percentile}_percentile_model_{model_num + 1}_in_run_{len(top_chosen_points[model_num])}_points.zip")
        current_model.save(model_save_path)
    # Log all the top percentile model information in a file
    with open(os.path.join(run_dir, f"ppo_silu_approx_top_{percentile}_percentile_model_info.txt"), 'w') as f:
        for model_num in range(len(top_models)):
            f.write(f"Model {model_num + 1}:\n")
            f.write(f"Reward: {top_rewards[model_num]}\n")
            f.write(f"Mean Error: {top_mean_errors[model_num]}\n")
            f.write(f"Max Error: {top_max_errors[model_num]}\n")
            f.write(f"Chosen Points: {top_chosen_points[model_num]}\n")
            f.write("\n")
    # Generate desensitized version of chosen points as commas in lists of numbers can cause problems in csv files
    # Convert commas to whitespace and the entire list to a string
    chosen_points_desensitized = [str(list_of_chosen_points[model_num]).replace(',', ' ') for model_num in range(len(list_of_chosen_points))]
    print(len(chosen_points_desensitized))
    print(chosen_points_desensitized)
    # Save all the reward, mean error, max error, and chosen points information in a csv file
    with open(os.path.join(run_dir, f"ppo_silu_approx_model_info.csv"), 'w') as f:
        f.write("Model Number,Reward,Mean Error,Max Error,Chosen Points\n")
        for model_num in range(len(list_of_models)):
            f.write(f"{model_num + 1},{list_of_rewards[model_num]},{list_of_mean_errors[model_num]},{list_of_max_errors[model_num]},{chosen_points_desensitized[model_num]}\n")




    # Generate plot of best model
    plot_chosen_points(silu, best_chosen_points, initial_range, show_plot=False, save_fig_name=os.path.join(run_dir, f"ppo_silu_approx_best_model_in_run_{len(best_chosen_points)}_points_{best_model_timestamp}.png"))

    histogram_bins = max(1,iter_num // 20)
    # Generate histogram of rewards
    plt.hist(list_of_rewards, bins=histogram_bins)
    plt.title('Histogram of Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(run_dir, f"ppo_silu_approx_best_model_in_run_histogram_of_rewards_{best_model_timestamp}.png"))
    plt.show()
    plt.close()
    # Generate histogram of mean errors
    plt.hist(list_of_mean_errors, bins=histogram_bins)
    plt.title('Histogram of Mean Errors')
    plt.xlabel('Mean Error')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(run_dir, f"ppo_silu_approx_best_model_in_run_histogram_of_mean_errors_{best_model_timestamp}.png"))
    plt.show()
    plt.close()
    # Generate histogram of max errors
    plt.hist(list_of_max_errors, bins=histogram_bins)
    plt.title('Histogram of Max Errors')
    plt.xlabel('Max Error')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(run_dir, f"ppo_silu_approx_best_model_in_run_histogram_of_max_errors_{best_model_timestamp}.png"))
    plt.show()
    plt.close()
    # Generate histogram of number of points
    plt.hist([len(x) for x in list_of_chosen_points], bins=histogram_bins)
    plt.title('Histogram of Number of Points')
    plt.xlabel('Number of Points')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(run_dir, f"ppo_silu_approx_best_model_in_run_histogram_of_number_of_points_{best_model_timestamp}.png"))
    plt.show()
    plt.close()

    # Histograms overlaid with graph being lines instead of bar graphs
    figure_legend_list = ['Rewards', 'Mean Errors', 'Max Errors']
    plot_overlapping_histogram(list_of_rewards, list_of_mean_errors, list_of_max_errors, figure_legend=figure_legend_list, figure_name=os.path.join(run_dir, f"ppo_silu_approx_best_model_in_run_histograms_overlaid_{best_model_timestamp}.png"))







if __name__ == '__main__':
    # Training run argument dictionary
    single_train_run_function_dictionary_silu = {
        'input_function_name': 'silu',
        'input_num_points': 8,
        'input_initial_range': (-8, 8),
        'input_function': silu,
        'input_curvature_function': silu_curvature_alt,
        'input_final_reward_function': final_reward_function_silu,
        'input_reward_function_definition': reciprocal_error_average_reward_function,
        'input_final_reward_error_function': final_reward_function_silu_print,
        'input_train_timesteps': 30_000,
        'input_environment': RL_Environment_test2_continuous_action_space_generalized_nonlinear_map_normalized_curvature,
        'input_verbose': False,
        'input_algorithm': PPO,
        'input_algorithm_name': 'PPO',
        'input_policy': 'MultiInputPolicy',
        'input_learning_rate': OneCycleLR_Schedule_Linear_MinLR,
        'learning_rate_configs': {'initial_value': 0.001, 'warmup_proportion': 0.1, 'multiplier': 20},
        'algorithm_parameters': {'ent_coef': 0.05, 'vf_coef': 0.5}
    }
    # Training run argument dictionary for sigmoid
    single_train_run_function_dictionary_sigmoid = {
        'input_function_name': 'sigmoid',
        'input_num_points': 8,
        'input_initial_range': (-8, 8),
        'input_function': sigmoid,
        'input_curvature_function': sigmoid_curvature,
        'input_reward_function_definition': reciprocal_error_average_reward_function,
        'input_final_reward_function': final_reward_function_sigmoid,
        'input_final_reward_error_function': final_reward_error_function_sigmoid,
        'input_train_timesteps': 30_000,
        'input_environment': RL_Environment_test2_continuous_action_space_generalized_nonlinear_map_normalized_curvature,
        'input_verbose': False,
        'input_algorithm': PPO,
        'input_algorithm_name': 'PPO',
        'input_policy': 'MultiInputPolicy',
        'input_learning_rate': OneCycleLR_Schedule_Cosine,
        'learning_rate_configs': {'initial_value': 0.001, 'warmup_proportion': 0.1, 'multiplier': 20},
        'algorithm_parameters': {'ent_coef': 0.05, 'vf_coef': 0.5}
    }
    # Training run argument dictionary for gelu
    single_train_run_function_dictionary_gelu = {
        'input_function_name': 'gelu',
        'input_num_points': 8,
        'input_initial_range': (-8, 8),
        'input_function': gelu,
        'input_curvature_function': gelu_curvature,
        'input_reward_function_definition': reciprocal_error_average_reward_function,
        'input_final_reward_function': final_reward_function_gelu,
        'input_final_reward_error_function': final_reward_error_function_gelu,
        'input_train_timesteps': 30_000,
        'input_environment': RL_Environment_test2_continuous_action_space_generalized_nonlinear_map_normalized_curvature,
        'input_verbose': False,
        'input_algorithm': PPO,
        'input_algorithm_name': 'PPO',
        'input_policy': 'MultiInputPolicy',
        'input_learning_rate': OneCycleLR_Schedule_Exponential,
        'learning_rate_configs': {'initial_value': 0.001, 'warmup_proportion': 0.1, 'multiplier': 20},
        'algorithm_parameters': {'ent_coef': 0.05, 'vf_coef': 0.5}
    }
    print('gelu Run:')
    for i in range(1):
        print(f'Gelu Run #{i+1}')
        final_reward, final_mean_error, final_max_error = \
            single_train_run_function(**single_train_run_function_dictionary_gelu)
    print('silu Run:')
    for i in range(1):
        print(f'Silu Run #{i+1}')
        final_reward, final_mean_error, final_max_error = \
            single_train_run_function(**single_train_run_function_dictionary_silu)
    print('sigmoid Run:')
    for i in range(1):
        print(f'Sigmoid Run #{i+1}')
        final_reward, final_mean_error, final_max_error = \
            single_train_run_function(**single_train_run_function_dictionary_sigmoid)
