import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from utils.basic_io import ensure_path

sns.set_style("whitegrid")

from mcts_general_experiments.mcts_experiments_api import collect_set_of_experiments


def get_total_reward_mean_low_and_high(experiment_df: pd.DataFrame):
    results_per_seed = pd.DataFrame()
    group_by_seed = experiment_df.groupby(['seed', 'num_simulations'])
    results_per_seed['sum'] = group_by_seed['reward'].sum()
    results_per_seed = results_per_seed.reset_index()
    group_by = results_per_seed.groupby('num_simulations')
    return group_by['sum'].mean(), group_by['sum'].min(), group_by['sum'].max()


def plot_total_reward(mean: pd.Series, low: pd.Series, high: pd.Series, title, ax: plt.Axes, yrange=None, label=None):
    # if title:
    #     ax.set_title(title, fontsize=10)
    # ax.set_xlabel('Num. Simulations', fontsize=9)
    # ax.set_ylabel('Total Reward        ', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    if yrange is not None:
        ax.set_ylim(yrange[0], yrange[1])
        ax.set_xlim(0, 1200)
    ax.plot(mean.index, mean.values, label=label)
    ax.fill_between(low.index, low.values, high.values, alpha=.3)


if __name__ == '__main__':
    plt.rc('pgf', texsystem='lualatex')
    plt.rcParams.update({'figure.autolayout': True})

    import matplotlib.pyplot as plt

    # plt.rcParams.update({
    #     "font.serif": [],  # use latex default serif font
    #     "font.sans-serif": [],  # use a specific sans-serif font
    # })


    for experiment_set in [['Pendulum-v0', 2., [-1500, -900]], ['ContinuousCartPoleEnv', 1., [0,201]]]:
        [env, sigma_factor, yrange] = experiment_set
        for sigma in [0.5, 0.75, 1.0]:
            sigma *= sigma_factor
            row = 0
            fig, axes = plt.subplots(nrows=4, ncols=4, sharex='all', sharey='all', figsize=(6, 4.5))
            fig.tight_layout()
            # fig.suptitle('{}, $\sigma={}$'.format(env, sigma))
            for c in [1, 2, 3, 4]:
                col = 0
                for alpha in [0.0625, 0.125, 0.25, 0.5]:
                    pad = 5
                    axes[row, 0].set_ylabel('Total Reward')
                    axes[row, 0].annotate('$C={}$'.format(c), xy=(0, 0.5), xytext=(-axes[row, 0].yaxis.labelpad - pad, 0),
                                xycoords=axes[row, 0].yaxis.label, textcoords='offset points',
                                ha='right', va='center')
                    axes[0, col].set_title('$\\alpha={}$'.format(alpha))
                    axes[-1, col].set_xlabel('Num. Simulations'.format(c))
                    path = "../result/{}/SPW_alpha_{:1.3f}_C_{}_Sigma_{}".format(env, alpha, c, sigma)
                    print(path)
                    experiment_df, _ = collect_set_of_experiments(path)
                    # try:
                    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
                    plot_total_reward(mean,
                                      low,
                                      high,
                                      title='$C={}, \\alpha={}$      '.format(c, alpha),
                                      ax=axes[row, col],
                                      yrange=yrange)
                    # except Exception as e:
                    #     pass

                    col += 1
                row += 1
            plt_path = '../plots/SPW_variation/{}_Sigma_{}.pgf'.format(env, sigma)
            ensure_path(plt_path)
            plt.savefig(plt_path)
    plt.show()

    # default pendulum plot
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharex='all')
    fig.tight_layout()
    path = '../result/Pendulum-v0/standard'
    ax = axes[0]
    ax.set_title('Pendulum-v0')
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Total Reward')
    ax.set_xlim(0, 3200)
    ax.set_ylim(-1600, -0)
    experiment_df, _ = collect_set_of_experiments(path)
    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high, path, ax, label='standard')

    # default cart pole plot
    ax = axes[1]
    path = '../result/CartPole-v0/standard'
    ax.set_title('CartPole-v0')
    ax.set_xlabel('Number of Simulations')
    # ax.set_ylabel('Total Reward')
    ax.set_xlim(0, 3200)
    ax.set_ylim(0, 201)
    experiment_df, _ = collect_set_of_experiments(path)
    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high, path, ax, label='standard')
    plt_path = '../plots/discrete/CartPole-v0-Pendulum_standard.pgf'
    ensure_path(plt_path)
    plt.savefig(plt_path)
    plt.show()

    # time discretization
    for with_skipping_in_acting in [True, False]:
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharex='all')
        fig.tight_layout()
        path = '../result/Pendulum-v0/standard'
        ax = axes[0]
        ax.set_title('Pendulum-v0')
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Total Reward')
        ax.set_xlim(0, 3200)
        ax.set_ylim(-1600, -0)
        experiment_df, _ = collect_set_of_experiments(path)
        mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
        plot_total_reward(mean, low, high, path, ax, label='standard')

        # default cart pole plot
        ax = axes[1]
        path = '../result/CartPole-v0/standard'
        ax.set_title('CartPole-v0')
        ax.set_xlabel('Number of Simulations')
        # ax.set_ylabel('Total Reward')
        ax.set_xlim(0, 3200)
        ax.set_ylim(0, 201)
        experiment_df, _ = collect_set_of_experiments(path)
        mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
        plot_total_reward(mean, low, high, path, ax, label='standard')

        ax.set_xlim(0, 100)
        for it in [2, 4, 8]:
            ax = axes[0]
            path = '../result/Pendulum-v0/time_step_skipping_x{}{}'.format(it, '_also_in_acting' if with_skipping_in_acting else '')
            experiment_df, _ = collect_set_of_experiments(path)
            mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
            plot_total_reward(mean, low, high, path, ax, label='{}x skipping'.format(it))

            ax = axes[1]
            path = '../result/CartPole-v0/time_step_skipping_x{}{}'.format(it, '_also_in_acting' if with_skipping_in_acting else '')
            experiment_df, _ = collect_set_of_experiments(path)
            mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
            plot_total_reward(mean, low, high, path, ax, label='{}x skipping'.format(it))
            ax.legend()

        plt_path = '../plots/discrete/CartPole-v0-Pendulum_time_skipping_{}.pgf'.format('in_acting' if with_skipping_in_acting else '')
        ensure_path(plt_path)
        plt.savefig(plt_path)
        plt.show()

    # Mountain Car with skipping
    fig, ax = plt.subplots(figsize=(2.5, 2.5), sharex='all')
    fig.tight_layout()
    path = '../result/MountainCar-v0/standard'
    # ax.set_title('MountainCar-v0')
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Total Reward')
    ax.set_xlim(0, 1600)
    ax.set_ylim(-201, -0)
    experiment_df, _ = collect_set_of_experiments(path)
    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high, path, ax, label='standard')

    for it in [2, 4, 8]:
        path = '../result/MountainCar-v0/time_step_skipping_x{}_also_in_acting'.format(it)
        experiment_df, _ = collect_set_of_experiments(path)
        mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
        plot_total_reward(mean, low, high, path, ax, label='{}x skipping'.format(it))
        ax.legend()

    plt_path = '../plots/discrete/MountainCar_time_skipping_also_in_acting.pgf'
    ensure_path(plt_path)
    plt.savefig(plt_path)
    plt.show()

    # Action Discretization
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    fig.tight_layout()
    ax = axes[0]
    path = '../result/Pendulum-v0/standard'
    ax.set_title('Pendulum-v0')
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Total Reward')
    ax.set_xlim(0, 3200)
    ax.set_ylim(-1600, -0)
    experiment_df, _ = collect_set_of_experiments(path)
    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high, path, ax, label='2 actions')

    # more actions
    path = '../result/Pendulum-v0/more_actions'
    experiment_df, _ = collect_set_of_experiments(path)
    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high, path, ax, label='4 actions')


    # default cart pole plot
    ax = axes[1]
    path = '../result/ContinuousCartPoleEnv/discrete_from_continuous'
    ax.set_title('ContinuousCartPoleEnv')
    ax.set_xlabel('Number of Simulations')
    # ax.set_ylabel('Total Reward')
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 201)
    experiment_df, _ = collect_set_of_experiments(path)
    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high, path, ax, label='standard')

    # more actions
    path = '../result/ContinuousCartPoleEnv/more_actions'
    experiment_df, _ = collect_set_of_experiments(path)
    mean, low, high = get_total_reward_mean_low_and_high(experiment_df)
    plot_total_reward(mean, low, high, path, ax, label='4 actions')

    ax.legend()

    plt_path = '../plots/discrete/CartPole-v0-Pendulum_more_actions.pgf'
    ensure_path(plt_path)
    plt.savefig(plt_path)
    plt.show()
