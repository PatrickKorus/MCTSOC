from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

from mcts_general_experiments.visualization import plot_total_reward
from q_learning_baseline.visualization import get_total_reward_mean_low_and_high_baseline
from utils.basic_io import ensure_path


def load_tf(path):
    """
    Taken from https://gist.github.com/willwhitney/9cecd56324183ef93c2424c9aa7a31b4
    """

    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    dframes = {}
    mnames = ea.Tags()['scalars']

    for n in mnames:
        dframes[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", n.replace('val/', '')])
        dframes[n].drop("wall_time", axis=1, inplace=True)
        dframes[n] = dframes[n].set_index("epoch")
    return pd.concat([v for k, v in dframes.items()], axis=1)

def get_mean_low_high_from_muzero_results(
        base_path='../result_muzero/default_cartpole_{}',
        x_column='2.Workers/2.Training steps',
        y_column='1.Total reward/1.Total reward'):
    df = pd.DataFrame()
    for seed in [68268, 42613, 43567]:
        tensorflow_data_per_seed = load_tf(base_path.format(seed))
        info = pd.DataFrame()
        info['x'] = tensorflow_data_per_seed[x_column]
        info['total_reward_{}'.format(seed)] = tensorflow_data_per_seed[y_column]
        if len(df) == 0:
            df = info
        else:
            df = df.merge(info, how='outer')

    df = df.sort_values('x').fillna(method='ffill').fillna(method='bfill')
    result_df = pd.DataFrame()
    for seed in [68268, 42613, 43567]:
        local_df = pd.DataFrame()
        local_df['total_reward'] = df['total_reward_{}'.format(seed)]
        local_df['seed'] = seed
        local_df['x'] = df['x']
        result_df = result_df.append(local_df)

    # hard fix for muzero adding 0 at the beginning where no evaluation took place yet..
    result_df = result_df[result_df['total_reward'] != 0]
    gb_time_step = result_df.groupby('x')
    mean = gb_time_step['total_reward'].mean()
    low = gb_time_step['total_reward'].min()
    high = gb_time_step['total_reward'].max()
    return mean, low, high


if __name__ == '__main__':
    plt.rc('pgf', texsystem='lualatex')
    plt.rcParams.update({'figure.autolayout': True})

    # get mean, low, high,
    # columns:
    # Index(['1.Total reward/1.Total reward', '1.Total reward/2.Mean value',
    #        '1.Total reward/3.Episode length', '1.Total reward/4.MuZero reward',
    #        '1.Total reward/5.Opponent reward', '2.Workers/1.Self played games',
    #        '2.Workers/2.Training steps', '2.Workers/3.Self played steps',
    #        '2.Workers/4.Reanalysed games',
    #        '2.Workers/5.Training steps per self played step ratio',
    #        '2.Workers/6.Learning rate', '3.Loss/1.Total weighted loss',
    #        '3.Loss/Value loss', '3.Loss/Reward loss', '3.Loss/Policy loss'],
    #       dtype='object')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 2.5))
    # mean, low, high = get_mean_low_high_from_muzero_results('../result_muzero/default_cartpole_{}',)
    # plot_total_reward(mean, low, high, 'muzero', ax)

    ax[0].set_xlabel('Number of Training Steps')
    ax[0].set_ylabel('Total Reward')
    ax[0].set_title('Pendulum-v0')
    mean, low, high = get_mean_low_high_from_muzero_results('../result_muzero/q_learning_pendulum_new_{}',)
    plot_total_reward(mean, low, high, 'muzero', ax[0], label='MuZero')
    mean, low, high = get_total_reward_mean_low_and_high_baseline(
        '../result_baseline/pendulum_20k_steps.pkl')
    plot_total_reward(mean, low, high, None, ax[0], label='DQN')
    ax[0].set_ylim(-1800, -199)
    ax[0].set_xlim(0, 20000)
    ax[1].set_xlabel('Number of Training Steps')
    ax[1].set_title('CartPole-v0')
    mean, low, high = get_mean_low_high_from_muzero_results('../result_muzero/q_learning_cartpole_{}',)
    plot_total_reward(mean, low, high, 'muzero', ax[1], label='MuZero')
    mean, low, high = get_total_reward_mean_low_and_high_baseline(
        '../q_learning_baseline/result_baseline/total_rewards_cartpole.pkl')
    plot_total_reward(mean, low, high, None, ax[1], label='DQN')
    ax[1].set_xlim(0, 5000)
    ax[1].set_ylim(0, 201)
    ax[0].legend()

    plt_path = '../plots/muzero/muzero_dqn.svg'
    ensure_path(plt_path)
    plt.savefig(plt_path)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 2.5))
    # mean, low, high = get_mean_low_high_from_muzero_results('../result_muzero/default_cartpole_{}',)
    # plot_total_reward(mean, low, high, 'muzero', ax)

    ax[0].set_xlabel('Number of Training Steps')
    ax[0].set_ylabel('Total Reward')
    ax[0].set_title('Pendulum-v0')
    mean, low, high = get_mean_low_high_from_muzero_results('../result_muzero/default_pendulum_new_{}',)
    plot_total_reward(mean, low, high, 'muzero', ax[0], label='MuZero')
    mean, low, high = get_total_reward_mean_low_and_high_baseline(
        '../result_baseline/pendulum_20k_steps.pkl')
    plot_total_reward(mean, low, high, None, ax[0], label='DQN')
    ax[0].set_ylim(-1800, -199)
    ax[0].set_xlim(0, 20000)
    ax[1].set_xlabel('Number of Training Steps')
    ax[1].set_title('CartPole-v0')
    mean, low, high = get_mean_low_high_from_muzero_results('../result_muzero/default_cartpole_{}',)
    plot_total_reward(mean, low, high, 'muzero', ax[1], label='MuZero')
    mean, low, high = get_total_reward_mean_low_and_high_baseline(
        '../q_learning_baseline/result_baseline/total_rewards_cartpole.pkl')
    plot_total_reward(mean, low, high, None, ax[1], label='DQN')
    ax[1].set_xlim(0, 5000)
    ax[1].set_ylim(0, 201)
    ax[0].legend()

    plt_path = '../plots/muzero/muzero_dqn_with_reananlyze.svg'
    ensure_path(plt_path)
    plt.savefig(plt_path)
    plt.show()
