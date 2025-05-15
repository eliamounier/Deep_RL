"""
Script to plot performance data from multiple algorithms on a given environment.
Usage:
Pendulum -> python plot.py --env pendulum --algo_names PPO TD3 SAC --colors '#CA027D' '#5CC9CA' '#7C67DD' --output_dir plots --title "Pendulum Performance" --show
Acrobot -> python plot.py --env acrobot --algo_names PPO DQN --colors '#CA027D' '#C59C6D' --output_dir plots --title "Acrobot Performance" --show
Cartpole -> python plot.py --env cartpole --algo_names PPO DQN --colors '#CA027D' '#C59C6D' --output_dir plots --title "Cartpole Performance" --show
Lunar Lander discrete -> python plot.py --env lunar-discrete --algo_names PPO DQN --colors '#CA027D' '#C59C6D' --output_dir plots --title "Discrete Lunar Lander Performance" --show
Lunar Lander continuous -> python plot.py --env lunar-continuous --algo_names PPO TD3 SAC --colors '#CA027D' '#5CC9CA' '#7C67DD' --output_dir plots --title "Continuous Lunar Lander Performance" --show
"""

import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_data(env, directory="plot_files_plk"):
    """
    Load pickle files containing timesteps, means, and standard deviations.
    Args:
        env (str): Environment name
        directory (str): Directory where the data is stored
    Returns:
        timesteps (list): List of timesteps
        means (list): List of mean rewards
        stds (list): List of standard deviations of rewards
    """
    timesteps_path = os.path.join(directory, env, "timesteps.pkl")
    means_path = os.path.join(directory, env, "means.pkl")
    stds_path = os.path.join(directory, env, "stds.pkl")

    with open(timesteps_path, "rb") as f:
        timesteps = pickle.load(f)
    with open(means_path, "rb") as f:
        means = pickle.load(f)
    with open(stds_path, "rb") as f:
        stds = pickle.load(f)

    return timesteps, means, stds


def create_plot(
    env,
    algo_names=None,
    colors=None,
    output_dir="plots",
    total_timesteps=None,
    title=None,
    show_plot=False,
):
    """ "
    Create plot from loaded data.
    Args:
        env (str): Environment name
        algo_names (list): List of algorithm names for the legend
        colors (list): List of colors for each algorithm (matching algo_names order)
        output_dir (str): Directory to save the plot
        total_timesteps (int): Total timesteps for filename
        title (str): Custom title for the plot
        show_plot (bool): Whether to show the plot after saving
    """
    os.makedirs(output_dir, exist_ok=True)

    all_timesteps, pooled_means, pooled_stds = load_data(env)

    if algo_names is None:
        algo_names = [f"Algorithm {i+1}" for i in range(len(pooled_means))]
    # If we have more data series than algorithm names, append generic names
    elif len(algo_names) < len(pooled_means):
        algo_names = algo_names + [
            f"Algorithm {i+1}" for i in range(len(pooled_means) - len(algo_names))
        ]
    # If we have more algorithm names than data series, truncate the list
    elif len(algo_names) > len(pooled_means):
        algo_names = algo_names[: len(pooled_means)]

    # Default colors if none provided
    if colors is None:
        colors = [
            "blue",
            "green",
            "red",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

    # Ensure we have enough colors (cycle through the provided ones if needed)
    if len(colors) < len(pooled_means):
        colors = colors * (len(pooled_means) // len(colors) + 1)

    # Truncate colors if we have more than needed
    colors = colors[: len(pooled_means)]

    plt.figure(figsize=(6.5, 4))
    for idx, timestep in enumerate(all_timesteps):
        plt.plot(timestep, pooled_means[idx], label=algo_names[idx], color=colors[idx])
        plt.fill_between(
            timestep,
            pooled_means[idx] - pooled_stds[idx],
            pooled_means[idx] + pooled_stds[idx],
            alpha=0.3,
            color=colors[idx],
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")

    if title:
        plt.title(title, fontweight="bold")
    else:
        plt.title(f"Performance of {env} on multiple algorithms")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    filename = (
        f"performance_{env}_{total_timesteps}.png"
        if total_timesteps
        else f"performance_{env}.png"
    )

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, transparent=True)
    print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Create plots from pickle data.")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=[
            "acrobot",
            "cartpole",
            "lunar-continuous",
            "lunar-discrete",
            "pendulum",
        ],
        help="Environment name",
    )
    parser.add_argument(
        "--algo_names",
        type=str,
        nargs="+",
        choices=["DQN", "PPO", "SAC", "TD3"],
        help="Algorithm names for legend",
    )
    parser.add_argument(
        "--colors",
        type=str,
        nargs="+",
        help="Colors for each algorithm (in the same order as algo_names)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Output directory for plots"
    )
    parser.add_argument(
        "--total_timesteps", type=int, help="Total timesteps for filename"
    )
    parser.add_argument("--title", type=str, help="Custom plot title")
    parser.add_argument(
        "--show", action="store_true", help="Show plot window after saving"
    )

    args = parser.parse_args()

    create_plot(
        env=args.env,
        algo_names=args.algo_names,
        colors=args.colors,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        title=args.title,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
