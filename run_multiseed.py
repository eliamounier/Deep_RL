import argparse
import gymnasium
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.vec_env import VecVideoRecorder

import os
import numpy
import matplotlib
import matplotlib.pyplot as plt

import random
import numpy as np
import torch
import tqdm

import pickle

# ENVS = ["MountainCar-v0", "CartPole", "MountainCar-v0", "MountainCarContinuous-v0", "Acrobot-v1", "Pendulum" ]
# LunarLanderContinuous-v3
## EASIEST AND QUICKEST: Lunar Lander, CartPole, Acrobot, and maybe pendulum -- I think mountaincar needs A LOT of steps


# python run.py --algo PPO --env MountainCar-v0 --num_timesteps 20000
def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument("--env", required=True, type=str, default="CartPole-v0", help="Environment name")
    parser.add_argument("--num-algos", required=True, type=int, help="Number of algorithms to use")

    args, remaining_argv = parser.parse_known_args()
    count = args.num_algos

    parser.add_argument("--algos", required=True, type=str, nargs=count, default="PPO", help="Algorithm to use.", choices=["PPO", "DQN", "TD3", "SAC"])
    parser.add_argument("--num-timesteps", type=int, default=10000, help="Number of timesteps to train")
    parser.add_argument("--seed", type=int, default=0, nargs=3, help="Random seed, must use 3")
    return parser.parse_args()

def get_solution_threshold(env):
    if env == 'LunarLander-v3':
        return 200
    elif env == 'Acrobot-v1':
        return -100
    elif env == 'Pendulum':
        return -0.5
    elif env == 'CartPole-v0':
        return 200
    elif env == 'CartPole-v1':
        return 500

def get_algorithm(algo_name):
    if algo_name == "PPO":
        return PPO
    elif algo_name == "SAC":
        return SAC
    elif algo_name == "TD3":
        return TD3
    elif algo_name == "DQN":
        return DQN

def train_model(env, algorithm, seed, timesteps, eval_freq, log_dir):
    solution_thresh = get_solution_threshold(env)

    train_env = gymnasium.make(env)
    val_env = make_vec_env(env, n_envs=1, seed=seed)

    eval_callback = EvalCallback(val_env,
                                best_model_save_path=log_dir,
                                log_path=log_dir,
                                eval_freq=eval_freq,
                                render=False,
                                deterministic=True,
                                n_eval_episodes=75)

    if algorithm == PPO :
        model = algorithm('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, seed=seed,
                        ent_coef=0.0, n_steps=2048, batch_size=256, gae_lambda=0.95,
                        gamma=0.99, n_epochs=10, clip_range=0.2, learning_rate=3e-4)
    elif algorithm == DQN:
        model = algorithm('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, seed=seed,
                        learning_rate=1e-3, buffer_size=50000,
                        batch_size=128, tau=1.0, gamma=0.99,
                        target_update_interval=1000, exploration_fraction=0.2,
                        exploration_final_eps=0.02)
    elif algorithm == TD3:
        model = algorithm('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, seed=seed,
                        learning_rate=1e-3, buffer_size=1000000, learning_starts=10000,
                        batch_size=100, tau=0.005, gamma=0.99, train_freq=(1, "episode"),
                        policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5)
    elif algorithm == SAC:
        model = algorithm('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, seed=seed,
                        learning_rate=3e-4, buffer_size=1000000, learning_starts=1000,
                        batch_size=256, tau=0.005, gamma=0.99, ent_coef='auto')


    model.learn(total_timesteps=timesteps,
                progress_bar=True,
                callback=eval_callback)

    train_env.close()
    val_env.close()

def get_log_dir(seed, env, algo_name):
    log_dir = f"./logs/seed_{seed}_{env}/{algo_name}/"
    return log_dir

def plot_avg_over_seeds(seeds, env, algo_name):
    timesteps = []
    results = []
    
    mean_results = []
    std_results = []

    for seed in seeds:
        log_dir = get_log_dir(seed, env, algo_name)

        # Load the evaluations.npz file
        data = numpy.load(os.path.join(log_dir, "evaluations.npz"))

        # Extract the relevant data
        timesteps.append(data['timesteps'])
        results.append(data['results'])

        # Calculate the mean and standard deviation of the results
        mean_results.append(numpy.mean(results[-1], axis=1))
        std_results.append(numpy.std(results[-1], axis=1))
    
    # Pooling the means and standard deviations
    print("result lengths")
    for result in mean_results:
        print(len(result))

    mean_results = np.array(mean_results)
    std_results = np.array(std_results)

    print("Mean results shape:\t", mean_results.shape)
    print("Std results shape:\t", std_results.shape)

    pooled_mean = np.mean(mean_results, axis = 0)
    print("Pooled mean shape:\t", pooled_mean.shape)

    pooled_std = np.mean(std_results, axis=0)
    print("Pooled std shape:\t", pooled_std.shape)

    seed_string = "_".join(str(seed) for seed in seeds)
    plot_log_dir = f"logs/seed_{seed_string}_{env}/{algo_name}"

    plt.figure()
    plt.plot(timesteps[-1], pooled_mean)
    plt.fill_between(timesteps[-1],
                                pooled_mean - pooled_std,
                                pooled_mean + pooled_std,
                                alpha=0.3)

    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title(f"{algo_name} Performance on {env}")

    if not os.path.exists(plot_log_dir):
        os.makedirs(plot_log_dir)

    plt.savefig(os.path.join(plot_log_dir, "performance.png"), dpi=300)

    return pooled_mean, pooled_std, timesteps


def main():
    args = parse_args()

    SEEDS = sorted(args.seed)  # or any integer
    print(SEEDS)

    ENV = args.env
    ALGO_NAMES = args.algos

    # ALGORITHM = get_algorithm(ALGO_NAMES)

    TOTAL_TIMESTEPS = args.num_timesteps
    EVAL_FREQUENCY = 20000
    
    pooled_means = []
    pooled_stds = []
    all_timesteps = []

    for algo in ALGO_NAMES:
        ALGORITHM = get_algorithm(algo)
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            log_dir = get_log_dir(seed, ENV, algo)

            # FROM HERE WE START WITH THE MODEL
            train_model(ENV, ALGORITHM, seed, TOTAL_TIMESTEPS, EVAL_FREQUENCY, log_dir)

        pooled_mean, pooled_std, timesteps = plot_avg_over_seeds(SEEDS, ENV, algo)

        pooled_means.append(pooled_mean)
        pooled_stds.append(pooled_std)
        all_timesteps.append(timesteps[0])

    seed_string = "_".join(str(seed) for seed in SEEDS)
    algo_joined = "_".join(alg for alg in ALGO_NAMES)

    plot_log_dir = f"logs/seed_{seed_string}_{ENV}"

    if not os.path.exists(plot_log_dir):
        os.makedirs(plot_log_dir)

    plt.figure()
    for idx, timestep in enumerate(all_timesteps):
        plt.plot(timestep, pooled_means[idx], label = ALGO_NAMES[idx])
        plt.fill_between(timestep,
                                    pooled_means[idx] - pooled_stds[idx],
                                    pooled_means[idx] + pooled_stds[idx],
                                    alpha=0.3)

    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title(f"Performance of {ENV} on multiple algorithms")
    plt.legend()
    plt.savefig(os.path.join(plot_log_dir, f"performance_{TOTAL_TIMESTEPS}_{algo_joined}.png"), dpi=300)

    os.makedirs(f'arrays/{ENV}', exist_ok=True)

    pickle.dump(pooled_means, open(f'arrays/{ENV}/means_{algo_joined}_{TOTAL_TIMESTEPS}.pkl', "wb"))
    pickle.dump(pooled_stds, open(f'arrays/{ENV}/stds_{algo_joined}_{TOTAL_TIMESTEPS}.pkl', "wb"))
    pickle.dump(all_timesteps, open(f'arrays/{ENV}/timesteps_{algo_joined}_{TOTAL_TIMESTEPS}.pkl', "wb"))

"""
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
"""
if __name__=="__main__":
    main()