import pandas as pd
import matplotlib.pyplot as plt

from mcts_general_experiments.visualization import plot_total_reward


def get_total_reward_mean_low_and_high_baseline(path):
    df = pd.read_pickle(path)
    gb = df.groupby('total_time_steps')
    mean, low, high = gb['total reward'].mean(), gb['total reward'].min(), gb['total reward'].max()
    return mean, low, high


if __name__ == '__main__':
    mean, low, high = get_total_reward_mean_low_and_high_baseline('result_baseline/total_rewards_full.pkl')
    fig, ax = plt.subplots()
    plot_total_reward(mean, low, high, None, ax)
    plt.show()