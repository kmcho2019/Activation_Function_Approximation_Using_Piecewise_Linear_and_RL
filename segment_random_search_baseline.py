import numpy as np
import matplotlib.pyplot as plt
from least_sq_approximation import combined_function_generator
from RL_PPO_SB3 import common_reward_function, plot_chosen_points
import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def plot_chosen_points_x_coord(original_function, chosen_points, initial_range, function_name='silu', show_plot=False,
                               save_fig_name=None):
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
    silu_reference_val = original_function(x)
    silu_approximation_val = piecewise_function(x)
    # visualize the function in matplotlib
    plt.plot(x, silu_reference_val, label=function_name)
    plt.plot(x, silu_approximation_val, label="approximation")
    plt.scatter(np.array(chosen_points), piecewise_function(np.array(chosen_points)), color='red',
                label=f"chosen points: {len(chosen_points)}")
    # Add x-coordinate text near the points
    for i, point in enumerate(chosen_points):
        plt.annotate(f"{point:.3f}", (point, piecewise_function(point)), xytext=(5, -10),
                     textcoords='offset points', ha='left', va='top')
    plt.legend()
    if show_plot:
        plt.show()
    if save_fig_name is not None:
        plt.savefig(save_fig_name)
    plt.close()


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
    if show_plot:
        plt.show()
    if save_fig_name is not None:
        plt.savefig(save_fig_name)
    plt.close()


# Given range [-A, A], number of points, function, search_iteration_number find the best solution with highest value
# that minimizes the error between the function and the piecewise linear approximation of the function
def segment_random_search_function(function, point_range, num_points, search_iteration_number=10000, is_sigmoid=False):
    best_reward = float('-inf')
    mean_error = None
    max_error = None
    best_points = []

    for _ in tqdm.tqdm(range(int(search_iteration_number))):
        random_points = sorted(np.random.uniform(low=point_range[0], high=point_range[1], size=num_points))
        step_size = 0.001  # determines the accuracy of the least square fit that generates the approximation of the silu function
        total_number_of_steps = int((point_range[1] - point_range[0]) / step_size) + 1
        # construct the piecewise linear approximation of the input function
        piecewise_function = combined_function_generator(function, random_points, total_number_of_steps, is_sigmoid=
        is_sigmoid)
        # compute the mean and max error of the approximation compared to the input function
        x = np.linspace(point_range[0], point_range[1], total_number_of_steps)
        reference_val = function(x)
        approximation_val = piecewise_function(x)
        mean_error = np.mean(np.abs(reference_val - approximation_val))
        max_error = np.max(np.abs(reference_val - approximation_val))
        # construct the reward function
        # the reward function takes the linear weighted sum of mean and max error and takes a form of a
        # monotonic decreasing function whose value is between 0 and 10
        # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
        reward = common_reward_function(mean_error, max_error)
        if reward > best_reward:
            best_reward = reward
            best_points = random_points
    return best_points, best_reward, mean_error, max_error


# random search except where the points have limited accuracy
def segment_random_search_function_in_discrete_interval(function, point_range, num_points, point_interval=0.01,
                                                        search_iteration_number=10000, is_sigmoid=False):
    best_reward = float('-inf')
    mean_error = None
    max_error = None
    best_points = []
    possible_points_size = int((point_range[1] - point_range[0]) / point_interval) + 1
    possible_points_array = np.linspace(point_range[0], point_range[1], possible_points_size)
    for _ in tqdm.tqdm(range(int(search_iteration_number))):
        # random_points = sorted(np.random.uniform(low=point_range[0], high=point_range[1], size=num_points))
        # choose random points in discrete array
        random_points = np.random.choice(possible_points_array, size=num_points, replace=False)
        # sort random points to be in ascending order
        random_points.sort()
        step_size = 0.001  # determines the accuracy of the least square fit that generates the approximation of the silu function
        total_number_of_steps = int((point_range[1] - point_range[0]) / step_size) + 1
        # construct the piecewise linear approximation of the input function
        piecewise_function = combined_function_generator(function, random_points, total_number_of_steps, is_sigmoid=
        is_sigmoid)
        # compute the mean and max error of the approximation compared to the input function
        x = np.linspace(point_range[0], point_range[1], total_number_of_steps)
        reference_val = function(x)
        approximation_val = piecewise_function(x)
        mean_error = np.mean(np.abs(reference_val - approximation_val))
        max_error = np.max(np.abs(reference_val - approximation_val))
        # construct the reward function
        # the reward function takes the linear weighted sum of mean and max error and takes a form of a
        # monotonic decreasing function whose value is between 0 and 10
        # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
        reward = common_reward_function(mean_error, max_error)
        if reward > best_reward:
            best_reward = reward
            best_points = random_points
    return best_points, best_reward, mean_error, max_error

# The segment points are symmetrical respect to the center of the interval
def segment_random_search_function_in_discrete_interval_symmetrical(function, point_range, num_half_points, point_interval=0.01,
                                                        search_iteration_number=10000, is_sigmoid=False):
    best_reward = float('-inf')
    mean_error = None
    max_error = None
    best_points = []

    # Exclude both ends of the partial_interval
    possible_points_array = np.arange(point_range[0] + point_interval, ((point_range[0] + point_range[1]) / 2), point_interval)
    for _ in tqdm.tqdm(range(int(search_iteration_number))):
        # random_points = sorted(np.random.uniform(low=point_range[0], high=point_range[1], size=num_points))
        # choose random points in discrete array
        partial_points = np.random.choice(possible_points_array, size=num_half_points, replace=False)
        # sort random points to be in ascending order
        partial_points.sort()
        # complete points to be symmetrical
        flipped_points = point_range[0] + point_range[1] - partial_points
        flipped_points.sort()
        random_points = np.concatenate((partial_points, flipped_points))
        step_size = 0.001  # determines the accuracy of the least square fit that generates the approximation of the silu function
        total_number_of_steps = int((point_range[1] - point_range[0]) / step_size) + 1
        # construct the piecewise linear approximation of the input function
        piecewise_function = combined_function_generator(function, random_points, total_number_of_steps, is_sigmoid=
        is_sigmoid)
        # compute the mean and max error of the approximation compared to the input function
        x = np.linspace(point_range[0], point_range[1], total_number_of_steps)
        reference_val = function(x)
        approximation_val = piecewise_function(x)
        mean_error = np.mean(np.abs(reference_val - approximation_val))
        max_error = np.max(np.abs(reference_val - approximation_val))
        # construct the reward function
        # the reward function takes the linear weighted sum of mean and max error and takes a form of a
        # monotonic decreasing function whose value is between 0 and 10
        # the reward function is constructed in such a way that the agent will be rewarded more if the error approaches 0
        reward = common_reward_function(mean_error, max_error)
        if reward > best_reward:
            best_reward = reward
            best_points = random_points
    return best_points, best_reward, mean_error, max_error

if __name__ == '__main__':
    iter_num = 10_000
    '''
    best_points, best_reward, mean_error, max_error = segment_random_search_function(function=silu, point_range=(-8, 8),
                                                                                     num_points=10,
                                                                                     search_iteration_number=iter_num)
    print('best points: ', best_points)
    print('best reward: ', best_reward)
    print('mean error: ', mean_error)
    print('max error: ', max_error)
    plot_chosen_points_x_coord(silu, best_points, (-8, 8), show_plot=True,
                               save_fig_name='silu_segment_random_search_10000_iterations.png')
    plot_chosen_points_x_coord_draft(original_function=silu, chosen_points=best_points, initial_range=(-8, 8), reward=
    best_reward, mean_error=mean_error, max_error=max_error, show_plot=True, save_fig_name=None)
    # best points:  [-5.796850806979917, -3.908373202615449, -1.2674294279956264, -0.35106220982126857, -0.31595593726923, 0.1531223679461693, 1.0706165585405678, 1.6957054701262155, 4.9784906137837766, 5.884760198450968]
    # best reward:  88.67887819890649
    # best points:  [-6.71374594540106, -3.3486526046123064, -3.0048314428649316, -1.0761795826371454, -0.4089874731456504, 0.07757272953379513, 0.6724982815810883, 1.1500455620703836, 5.19767989865184, 6.513842717424355]
    # best reward:  89.90661407829195
    # best points: [-6.315893071134241, -2.8032967349396003, -1.8232675033878962, -1.006057111879322, -0.2120619991821897,
    #          0.4491267514083024, 1.278752993426595, 4.281812198994521, 6.2556145099206635, 7.585135072254078]
    # best reward: 89.85106155981242
    # mean error: 0.019112772015932495
    # max error: 0.11406023833996501

    # discrete interval, step size 0.01 (same resolution as RL)
    best_points, best_reward, mean_error, max_error = segment_random_search_function_in_discrete_interval(function=silu,
                                                                                                          point_range=(
                                                                                                          -8, 8),
                                                                                                          num_points=10,
                                                                                                          search_iteration_number=iter_num)
    print('best points: ', best_points)
    print('best reward: ', best_reward)
    print('mean error: ', mean_error)
    print('max error: ', max_error)
    plot_chosen_points_x_coord(silu, best_points, (-8, 8), show_plot=True,
                               save_fig_name='silu_segment_random_search_discrete_steps_0.01_10000_iterations.png')
    plot_chosen_points_x_coord_draft(original_function=silu, chosen_points=best_points, initial_range=(-8, 8), reward=
    best_reward, mean_error=mean_error, max_error=max_error, show_plot=True, save_fig_name=None)
    # best points:  [-6.8  -2.98 -2.09 -1.03 -0.84 -0.02  0.61  1.43  4.62  5.69]
    # best reward:  89.47963441915222
    # mean error:  0.02355894691338284
    # max error:  0.19080689715825394
    '''

    # Attempt this for sigmoid
    # output = segment_random_search_function(function=sigmoid, point_range=(-8, 8), num_points=9,
    #                                         search_iteration_number=10_000, is_sigmoid=True)
    output = segment_random_search_function_in_discrete_interval_symmetrical(function=sigmoid, point_range=(-8, 8),
                                                                             num_half_points=4,
                                                                             search_iteration_number=iter_num,
                                                                             is_sigmoid=True)
    best_points, best_reward, mean_error, max_error = output
    print('best points: ', best_points)
    print('best reward: ', best_reward)
    print('mean error: ', mean_error)
    print('max error: ', max_error)

    plot_chosen_points_x_coord_draft(original_function=sigmoid, chosen_points=best_points, initial_range=(-8, 8),
                                     reward=best_reward, mean_error=mean_error, max_error=max_error,
                                     function_name='sigmoid', show_plot=True,
                                     save_fig_name='sigmoid_segment_random_search_10000_iterations.png')
    # best points:  [-4.96 -2.83 -1.85 -0.96  0.96  1.85  2.83  4.96]
    # best reward:  96.7448173619822
    # mean error:  0.012046741274713203
    # max error:  0.05719430017659623
