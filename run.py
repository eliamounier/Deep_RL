import argparse
import gymnasium
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder

import os
import numpy
import matplotlib
import matplotlib.pyplot

import random
import numpy as np
import torch




# ENVS = ["MountainCar-v0", "CartPole", "MountainCar-v0", "MountainCarContinuous-v0", "Acrobot-v1", "Pendulum" ]
# LunarLanderContinuous-v3
## EASIEST AND QUICKEST: Lunar Lander, CartPole, Acrobot, and maybe pendulum -- I think mountaincar needs A LOT of steps


# python run.py --algo PPO --env MountainCar-v0 --num_timesteps 20000
def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Environment name")
    parser.add_argument("--algo", type=str, default="PPO", help="Algorithm to use")
    parser.add_argument("--num_timesteps", type=int, default=10000, help="Number of timesteps to train")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()

args = parse_args()

SEED = args.seed  # or any integer
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
ENV = args.env
ALGO_NAME = args.algo

if ALGO_NAME == "PPO":
    ALGORITHM = PPO
elif ALGO_NAME == "SAC":
    ALGORITHM = SAC
elif ALGO_NAME == "TD3":
    ALGORITHM = TD3
elif ALGO_NAME == "DQN":
    ALGORITHM = DQN

TOTAL_TIMESTEPS = args.num_timesteps
EVAL_FREQUENCY = TOTAL_TIMESTEPS//20 #50 is the number of times we'll eval


log_dir = f"./logs/seed_{SEED}_{ENV}/{ALGO_NAME}/"    

env = gymnasium.make(ENV)


env_val = make_vec_env(ENV, n_envs=1, seed=SEED)
eval_callback = EvalCallback(env_val,
                            best_model_save_path=log_dir,
                            log_path=log_dir,
                            eval_freq=EVAL_FREQUENCY,
                            render=False,
                            deterministic=True,
                            n_eval_episodes=15)

if ALGORITHM == PPO :
    model = ALGORITHM('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, seed=SEED, ent_coef=0.05)
else: 
    model = ALGORITHM('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, seed=SEED)


model.learn(total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True,
            callback=eval_callback, 
            log_interval=2)


print("RUNNING EVAL...")
model.save(os.path.join(log_dir, f"_{ALGO_NAME}_{ENV}"))
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("DONE")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
env_val.close()

# Load the evaluations.npz file
data = numpy.load(os.path.join(log_dir, "evaluations.npz"))

# Extract the relevant data
timesteps = data['timesteps']
results = data['results']

# Calculate the mean and standard deviation of the results
mean_results = numpy.mean(results, axis=1)
std_results = numpy.std(results, axis=1)

# Plot the results
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(timesteps, mean_results)
matplotlib.pyplot.fill_between(timesteps,
                            mean_results - std_results,
                            mean_results + std_results,
                            alpha=0.3)

matplotlib.pyplot.xlabel('Timesteps')
matplotlib.pyplot.ylabel('Mean Reward')
matplotlib.pyplot.title(f"{ALGO_NAME} Performance on {ENV}")
matplotlib.pyplot.savefig(os.path.join(log_dir, "performance.png"))

# Create Elevation environment

env = make_vec_env(ENV, n_envs=1, seed=SEED)

# Load the best model
best_model_path = os.path.join(log_dir, "best_model.zip")
best_model = ALGORITHM.load(best_model_path, env=env)

print("RUNNING EVAL ON BEST MODEL...")
mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=100)
print("DONE")
print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Record video of the best model playing Lunar Lander
env = VecVideoRecorder(env, "./videos/",
                    video_length=15000,
                    record_video_trigger=lambda x: x == 0,
                    name_prefix=f"best_model_{ENV}_{ALGO_NAME}_seed_{SEED}")

obs = env.reset()
for _ in range(15000):
    action, _states = best_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:        
        break

env.close()
