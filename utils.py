import json
import matplotlib.pyplot as plt
import numpy as np


def load_stats(path):
    with open(path, "r") as f:
        stats = json.load(f)

    max_score = 0
    stats["max_scores"] = []
    for score in stats["max_scores"]:
        max_score = max(score, max_score)
        stats["max_scores"].append(max_score)

    return stats


def plot_stats(stats, log_y_axis=False):
    fig, ax = plt.subplots()
    plt.ylabel("Score")
    plt.xlabel("Episode")
    if log_y_axis:
        ax.set_yscale("log")
    plt.scatter(stats["episodes"], stats["scores"], label="score")
    plt.plot(stats["episodes"], stats["max_scores"], label="max_score")
    plt.plot(stats["episodes"], np.convolve(stats["scores"], np.ones((50,)) / 50, mode="same", label="mean_score"))

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    stats = load_stats("saved/q_learning/training_data/training_data_episode_77000.json")
    plot_stats(stats)
