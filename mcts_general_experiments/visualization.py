import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

from mcts_general_experiments.mcts_experiments_api import collect_set_of_experiments


def get_total_reward_mean_low_and_high(experiment_df: pd.DataFrame):
    results_per_seed = pd.DataFrame()
    group_by_seed = experiment_df.groupby(['seed', 'num_simulations'])
    results_per_seed['sum'] = group_by_seed['reward'].sum()
    results_per_seed = results_per_seed.reset_index()
    group_by = results_per_seed.groupby('num_simulations')
    return group_by['sum'].mean(), group_by['sum'].min(), group_by['sum'].max()


def plot_total_reward(mean: pd.Series, low: pd.Series, high: pd.Series, title, ax: plt.Axes, yrange=None):
    ax.set_title(title)
    ax.set_xlabel('Number of Simulations per Step')
    ax.set_ylabel('Total Reward')
    if yrange is not None:
        ax.set_ylim(yrange[0], yrange[1])
        ax.set_xlim(0, 1000)
    ax.plot(mean.index, mean.values)
    ax.fill_between(low.index, low.values, high.values, alpha=.3)


if __name__ == '__main__':
    # fig, axes = plt.subplots(nrows=3, ncols=3)
    # row = 0
    # for c in [1, 2, 3]:
    #     col = 0
    #     for alpha in [0.125, 0.25, 0.5]:
    #         path = '../result/Pendulum-v0/SPW_alpha_{:1.3f}_C_{}'.format(alpha, c)
    #         experiment_df, _ = collect_set_of_experiments(path)
    #         try:
    #             mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    #             plot_total_reward(mean, low, high, 'C = {}, alpha = {}'.format(c, alpha), axes[row, col])
    #         except Exception as e:
    #             pass
    #
    #         col += 1
    #     row += 1
    fig, ax = plt.subplots()
    path = '../result/ContinuousCartPoleEnv_old_new/discrete_from_continuous'
    experiment_df, _ = collect_set_of_experiments(path)
    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high, path, ax)
    plt.show()
