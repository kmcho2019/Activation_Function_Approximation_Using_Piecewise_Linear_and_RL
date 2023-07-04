import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import datetime
import os
import tqdm
import time


from least_sq_approximation import matrix_row_generator
from least_sq_approximation import piece_wise_linear_function_estimator
from least_sq_approximation import piecewise_linear_function_constructor
from least_sq_approximation import piecewise_linear_function_constructor_vectorized
from least_sq_approximation import combined_function_generator



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

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


# serves as the common reward function for final_reward_function_silu and final_reward_function_silu_print
def common_reward_function(mean_error, max_error):
    weighted_error = 0.2 * mean_error + 0.8 * max_error
    reward = 100 * np.exp(-weighted_error * 5)
    return reward

# takes in the list of all points chosen by the agent and the initial range
# then it constructs a piecewise linear approximation of the silu function
# it computes the mean and max error of the approximation compared to the silu function
# with the mean and max error it constructs a reward function that is used by the agent
def final_reward_function_silu(chosen_points, initial_range):
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
    reward = common_reward_function(mean_error, max_error)

    return reward

# takes in the list of all points chosen by the agent and the initial range
# and prints the values of the reward and error
def final_reward_function_silu_print(chosen_points, initial_range, verbose = True):
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
    reward = common_reward_function(mean_error, max_error)

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

# Given a function and a list of points, this function will plot the function and the piecewise linear approximation based on the points
# And also mark the points on the plot
def plot_chosen_points(original_function, chosen_points, initial_range, show_plot = False, save_fig_name = None):
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
    plt.plot(x, silu_reference_val, label="silu")
    plt.plot(x, silu_approximation_val, label="approximation")
    plt.scatter(np.array(chosen_points), piecewise_function(np.array(chosen_points)), color='red',
                label=f"chosen points: {len(chosen_points)}")
    plt.legend()
    if show_plot:
        plt.show()
    if save_fig_name is not None:
        plt.savefig(save_fig_name)
    plt.close()
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
            reward = -100

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


# Training function
def train_ppo(test_enabled=True, initial_range=(-8, 8), num_points=10, learning_rate=0.0002, train_timesteps= 10_000, save_model=False, verbose=True):
    # Create a vectorized environment with custom range and number of points

    env = DummyVecEnv([lambda: RL_Environment(initial_range=initial_range, num_points=num_points)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # Initialize the PPO model
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=int(verbose), tensorboard_log= './SiLU_approx_tensorboard_logs/')

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

# function for giving overlapping histogram
import matplotlib.pyplot as plt
import numpy as np

def plot_overlapping_histogram(A, B, C, figure_legend=None, figure_name=None):
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

    # Plot dots for each dataset
    plt.scatter(A, np.zeros_like(A), color='blue', label=figure_legend[0] if figure_legend else None)
    plt.scatter(B, np.zeros_like(B), color='orange', label=figure_legend[1] if figure_legend else None)
    plt.scatter(C, np.zeros_like(C), color='green', label=figure_legend[2] if figure_legend else None)

    if figure_name is not None:
        # Save the figure
        plt.savefig(figure_name)

    # Display the plot
    plt.show()
    plt.close()


if __name__ == '__main__':
    initial_range = (-8, 8)

    # Start the training
    model, final_chosen_points, reward, mean_error, max_error = train_ppo(initial_range=initial_range)

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

    # add tqdm tracking over iterations

    for i in tqdm.tqdm(range(iter_num)):
        model, final_chosen_points, reward, mean_error, max_error = train_ppo(initial_range=initial_range, test_enabled=True, verbose=False)
        list_of_rewards.append(reward)
        list_of_mean_errors.append(mean_error)
        list_of_max_errors.append(max_error)
        list_of_chosen_points.append(final_chosen_points)
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

    # Generate plot of best model
    plot_chosen_points(silu, best_chosen_points, initial_range, show_plot=False, save_fig_name=f"ppo_silu_approx_best_model_in_run_{len(best_chosen_points)}_points_{best_model_timestamp}.png")

    histogram_bins = iter_num // 5
    # Generate histogram of rewards
    plt.hist(list_of_rewards, bins=histogram_bins)
    plt.title('Histogram of Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join('model_archive', f"ppo_silu_approx_best_model_in_run_histogram_of_rewards_{best_model_timestamp}.png"))
    plt.show()
    plt.close()
    # Generate histogram of mean errors
    plt.hist(list_of_mean_errors, bins=histogram_bins)
    plt.title('Histogram of Mean Errors')
    plt.xlabel('Mean Error')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join('model_archive', f"ppo_silu_approx_best_model_in_run_histogram_of_mean_errors_{best_model_timestamp}.png"))
    plt.show()
    plt.close()
    # Generate histogram of max errors
    plt.hist(list_of_max_errors, bins=histogram_bins)
    plt.title('Histogram of Max Errors')
    plt.xlabel('Max Error')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join('model_archive', f"ppo_silu_approx_best_model_in_run_histogram_of_max_errors_{best_model_timestamp}.png"))
    plt.show()
    plt.close()
    # Generate histogram of number of points
    plt.hist([len(x) for x in list_of_chosen_points], bins=histogram_bins)
    plt.title('Histogram of Number of Points')
    plt.xlabel('Number of Points')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join('model_archive', f"ppo_silu_approx_best_model_in_run_histogram_of_number_of_points_{best_model_timestamp}.png"))
    plt.show()
    plt.close()

    # Histograms overlaid with graph being lines instead of bar graphs
    figure_legend_list = ['Rewards', 'Mean Errors', 'Max Errors']
    plot_overlapping_histogram(list_of_rewards, list_of_mean_errors, list_of_max_errors, figure_legend=figure_legend_list, figure_name=os.path.join('model_archive', f"ppo_silu_approx_best_model_in_run_histograms_overlaid_{best_model_timestamp}.png"))
