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


def plot_total_reward(mean: pd.Series, low: pd.Series, high: pd.Series):
    fig, ax = plt.subplots()
    ax.plot(mean.index, mean.values)
    ax.fill_between(low.index, low.values, high.values, alpha=.3)
    plt.show()


if __name__ == '__main__':
    path = '../result/ContinuousCartPoleEnv/discrete_from_continuous'
    experiment_df, config = collect_set_of_experiments(path)

    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high)
