import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from RL_PPO_SB3 import RL_Environment


from RL_PPO_SB3 import final_reward_function_silu_print

def show_steps_of_model(model, env, n_steps):
    obs = env.reset()

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        print("Action: ", action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print('Observation: ', obs)
        print('Reward: ', reward)
        print('Done: ', done)
        print('Chosen Points: ', env.envs[0].chosen_points)
        if done:
            print("Goal reached!", "reward=", reward)
            final_reward_function_silu_print(env.envs[0].chosen_points, env.envs[0].initial_range)
            break

initial_range = (-8, 8)
num_points = 10

env = DummyVecEnv([lambda: RL_Environment(initial_range=(-8, 8), num_points=10)])
eval_environment = VecNormalize(env, norm_obs=True, norm_reward=False)

# Random Agent, before training
random_model = PPO("MlpPolicy", eval_environment, verbose=1)
mean_reward, std_reward = evaluate_policy(random_model, eval_environment, n_eval_episodes=10)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
show_steps_of_model(random_model, eval_environment, num_points)
# eval_model = PPO.load('dqn_model.zip')
eval_model = PPO.load('ppo_model.zip')


# Evaluate trained agent
mean_reward, std_reward = evaluate_policy(eval_model, eval_environment, n_eval_episodes=10)
show_steps_of_model(eval_model, eval_environment, num_points)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
