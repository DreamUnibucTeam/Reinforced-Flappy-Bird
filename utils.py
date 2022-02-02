import json
import matplotlib.pyplot as plt
import numpy as np

def add_max_scores(stats):
    max_score = 0
    stats["max_scores"] = []
    for score in stats["scores"]:
        max_score = max(score, max_score)
        stats["max_scores"].append(max_score)
    
    return stats

def load_stats(path):
    with open(path, "r") as f:
        stats = json.load(f)

    return add_max_scores(stats)


def plot_stats(stats, log_y_axis=False):
    fig, ax = plt.subplots()
    plt.ylabel("Score")
    plt.xlabel("Episode")
    if log_y_axis:
        ax.set_yscale("log")
    plt.scatter(stats["episodes"], stats["scores"], label="score")
    plt.plot(stats["episodes"], stats["max_scores"], label="max_score")
    plt.plot(stats["episodes"], np.convolve(stats["scores"], np.ones((50,)) / 50, mode="same"), label="mean_score")

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.legend(loc="upper left")
    fig.tight_layout()
    plt.show()

def generate_for_dqn():
    too_good = load_stats("saved/dqn/result_525000.json")
    too_bad = load_stats('saved/dqn/result_250000.json')

    # combine these two
    stats = dict()
    stats['episodes'] = []
    stats['scores'] = []

    for i in range(0, len(too_good['episodes'])):
        stats['episodes'].append(i)
        stats['scores'].append(too_good['scores'][i])

    for i in range(len(too_bad['episodes'])):
        stats["episodes"].append(len(too_good['episodes']) + i)
        stats["scores"].append(too_bad['scores'][i])

    stats = add_max_scores(stats)
    print(len(stats['episodes']), len(stats['scores']), len(stats['max_scores']))
    plot_stats(stats)

if __name__ == "__main__":
    bad_data = load_stats('saved/q_learning/training_data_best/training_data_episode_60000.json')
    plot_stats(bad_data)
    
