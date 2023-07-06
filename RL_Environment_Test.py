# Test File, used to check if the RL_Environment is set up correctly

from RL_PPO_SB3 import RL_Environment, final_reward_function_silu, silu_curvature_alt
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
from datetime import datetime


class RL_Environment_test(gym.Env):
    def __init__(self, initial_range=(-8, 8), num_points=10, action_increment=0.01):
        super(RL_Environment_test, self).__init__()
        self.initial_range = initial_range
        self.range = self.initial_range
        self.total_points = num_points
        self.points_left = self.total_points
        self.chosen_points = []

        # Change action space to be Discrete
        self.num_actions = int((self.initial_range[1] - self.initial_range[0]) / action_increment) + 1
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # Define the observation space as a Tuple space
        num_points_space = gym.spaces.Discrete(self.total_points + 1)
        last_point_space = gym.spaces.Box(low=np.array([self.initial_range[0]]), high=np.array([self.initial_range[1]]),
                                          dtype=np.float32)
        space_to_end_space = gym.spaces.Box(low=np.array([0.]),
                                            high=np.array([self.initial_range[1] - self.initial_range[0]]),
                                            dtype=np.float32)

        self.observation_space = gym.spaces.Tuple((num_points_space, last_point_space, space_to_end_space))

    def reset(self, seed=None, **kwargs):
        self.range = self.initial_range
        self.points_left = self.total_points
        self.chosen_points = []
        # return value from reset should be: observation, info
        return (self.points_left, np.array([self.range[0]], dtype=np.float32),
                np.array([self.initial_range[1] - self.initial_range[0]], dtype=np.float32)), {}

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
            return (self.points_left, np.array([self.range[0]], dtype=np.float32),
                    np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)), reward, False, True, {}

        # Check if all points are used up
        if self.points_left == 0:
            # Calculate final reward and reward for the chosen point
            final_reward = final_reward_function_silu(self.chosen_points, self.initial_range)
            reward_for_choosing_point = silu_curvature_alt(chosen_point)
            combined_reward = final_reward + reward_for_choosing_point
            # return value from step should be: observation, reward, terminated, truncated, info
            return (self.points_left, np.array([self.range[0]], dtype=np.float32),
                    np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)), combined_reward, True, False, {}
        else:
            # Calculate reward for the chosen point
            reward = silu_curvature_alt(chosen_point)
            # return value from step should be: observation, reward, terminated, truncated, info
            return (self.points_left, np.array([self.range[0]], dtype=np.float32),
                    np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)), reward, False, False, {}


class RL_Environment_test2(gym.Env):
    def __init__(self, initial_range=(-8, 8), num_points=10, action_increment=0.01):
        super(RL_Environment_test2, self).__init__()
        self.initial_range = initial_range
        self.range = self.initial_range
        self.total_points = num_points
        self.points_left = self.total_points
        self.chosen_points = []

        # Change action space to be Discrete
        self.num_actions = int((self.initial_range[1] - self.initial_range[0]) / action_increment) + 1
        self.action_space = gym.spaces.Discrete(self.num_actions)

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
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, reward, False, True, {}

        # Check if all points are used up
        if self.points_left == 0:
            # Calculate final reward and reward for the chosen point
            final_reward = final_reward_function_silu(self.chosen_points, self.initial_range)  # this function should be defined
            reward_for_choosing_point = silu_curvature_alt(chosen_point)  # this function should be defined
            combined_reward = final_reward + reward_for_choosing_point
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, combined_reward, True, False, {}
        else:
            # Calculate reward for the chosen point
            reward = silu_curvature_alt(chosen_point)  # this function should be defined
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, reward, False, False, {}


class RL_Environment_test2_continuous_action_space(gym.Env):
    def __init__(self, initial_range=(-8, 8), num_points=10, action_increment=0.01):
        super(RL_Environment_test2_continuous_action_space, self).__init__()
        self.initial_range = initial_range
        self.range = self.initial_range
        self.total_points = num_points
        self.points_left = self.total_points
        self.chosen_points = []

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
            final_reward = final_reward_function_silu(self.chosen_points, self.initial_range)  # this function should be defined
            reward_for_choosing_point = silu_curvature_alt(chosen_point)  # this function should be defined
            combined_reward = final_reward + reward_for_choosing_point
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, combined_reward, True, False, {}
        else:
            # Calculate reward for the chosen point
            reward = silu_curvature_alt(chosen_point)  # this function should be defined
            # return value from step should be: observation, reward, terminated, truncated, info
            return {
                'num_points_left': self.points_left,
                'last_chosen_point': np.array([self.range[0]], dtype=np.float32),
                'space_to_end': np.array([self.initial_range[1] - self.range[0]], dtype=np.float32)
            }, reward, False, False, {}

# # print('Testing RL_Environment_test')
# # # env = RL_Environment()
# # env = RL_Environment_test()
# # # If the environment don't follow the interface, an error will be thrown
# # check_env(env, warn=True)
#
# print('Testing RL_Environment_test2')
# env = RL_Environment_test2()
# # If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)
#
# # env = RL_Environment()
# # check_env(env, warn=True)
# print('Testing RL_Environment_test2_continuous_action_space')
# env = RL_Environment_test2_continuous_action_space()
# check_env(env, warn=True)

# import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from RL_PPO_SB3 import final_reward_function_silu_print, plot_chosen_points, silu

# Define the custom environment class here (as you provided) ...
# RL_Environment_test2_continuous_action_space

# Register the custom environment
gym.envs.register(
    id="RLTestEnv-v0",
    entry_point=RL_Environment_test2_continuous_action_space,
)

# Make the custom environment
env = gym.make("RLTestEnv-v0")

# Wrap the environment with the Monitor wrapper
env = Monitor(env)

# As Stable Baselines 3 requires the environment to be vectorized, we can use DummyVecEnv for single environments
env = DummyVecEnv([lambda: env])

# Instantiate the agent
# Note: You might need to modify the policy architecture or use different hyperparameters for better performance.
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./test_environment_tensorboard_log/")

# Train the agent
training_total_timesteps = 100_000
model.learn(total_timesteps=training_total_timesteps,
            tb_log_name=f'PPO_modified_env_run_timesteps{training_total_timesteps}',progress_bar=True)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Get timestamp for the model name
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"test_environment_model_{time_stamp}_PPO_mean_reward_{mean_reward:.2f}"
model.save(model_name)

n_steps = 10

obs = env.reset()
verbose = True

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
        plot_chosen_points(silu, final_chosen_points, initial_range=env.envs[0].initial_range, show_plot=True)
        break
# 10_000 steps
# mean error:  0.3532070240361596
# max error:  1.6666231272869934
# reward:  0.08940940981001305
# Final Chosen Points:  [-7.340541124343872, -6.6700615882873535, -6.086956739425659, -5.460658311843872, -4.909785747528076, -4.482234239578247, -3.9684042930603027, -3.627002477645874, -3.333117961883545, -3.01991868019104]