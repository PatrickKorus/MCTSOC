import os
from logging import warning

from datetime import time, date, datetime

import gym
import typing
from copy import deepcopy

from mcts_general.agent import ContinuousMCTSAgent, MCTSAgent
from mcts_general.config import MCTSContinuousAgentConfig, MCTSAgentConfig
from mcts_general.game import ContinuousGymGame, DeepCopyableGame

import numpy as np
import pandas as pd

from utils.basic_io import ensure_path, list_files, dump, load


class MCTSExperiment:

    def __init__(self, game: DeepCopyableGame, config: MCTSAgentConfig, continuous: bool):
        self.game = game

        if continuous:
            assert isinstance(game, ContinuousGymGame), 'Game must be continuous!'
            assert isinstance(config, MCTSContinuousAgentConfig), 'Config must be suited for continuous agent!'
            self.agent = ContinuousMCTSAgent(config)
        else:
            self.agent = MCTSAgent(config)

    def step(self, state, reward, done, output_debug_info=False):
        action, info = self.agent.step(self.game, state, reward, done, True)
        state, reward, done = self.game.step(action)
        if output_debug_info:
            return state, reward, done, info
        else:
            return state, reward, done

    def run(self, render=False):
        done = False
        state = self.game.reset()
        reward = 0
        result = []
        while not done:
            state, reward, done, info = self.step(state, reward, done, True)
            if render:
                self.game.render()
            result.append({'state': state, 'reward': reward, 'done': done, 'max_tree_depth': info['max_tree_depth']})
        return pd.DataFrame(result)


def run_set_of_experiments(
        tag,
        game_constructor,
        default_config: MCTSAgentConfig,
        num_of_seeds: int,
        continuous: bool,
        num_simulations_list: typing.List[int] = [10, 50, 100, 200, 400, 800, 1600, 3200]
):
    rand = np.random
    rand.seed(0)
    seeds = rand.randint(0, 1e5, size=num_of_seeds)
    game = game_constructor(0)
    path = '{}/{}'.format(game, tag)
    ensure_path(path)
    dump('{}/config.dump'.format(path), default_config)

    for num_sim in num_simulations_list:
        for seed in seeds:
            print('starting experiment', tag, num_sim, seed)
            tic = datetime.now()
            config = deepcopy(default_config)
            config.num_simulations = num_sim
            game = game_constructor(int(seed))
            experiment = MCTSExperiment(game, config, continuous)

            dump_target = '{}/{}_{}.dump'.format(path, seed, num_sim)
            if os.path.isfile(dump_target):
                print("{} exists. Skipping...".format(dump_target))
                continue

            result = experiment.run()
            result['seed'] = seed
            result['num_simulations'] = num_sim
            result.to_pickle(dump_target)
            toc = datetime.now()
            print('done, time: ', toc - tic)


def collect_set_of_experiments(
        path,
        file_type='.dump'):

    files = list_files(path, file_type=file_type)
    df = pd.DataFrame()
    for file in files:
        try:
            if file == 'config.dump':
                config = load(path + '/' + file)
            else:
                new_df = pd.read_pickle(path + '/' + file)
                df = df.append(new_df)
        except EOFError as e:
            warning("{} couldn't be read".format(file))
    return df, config


if __name__ == '__main__':
    config = MCTSContinuousAgentConfig()
    config.do_roll_outs = False

    game_const = lambda seed: ContinuousGymGame(env=gym.make('Pendulum-v0'), seed=seed, mu=1, sigma=1.5)

    run_set_of_experiments(
        tag='initial_test',
        game_constructor=game_const,
        default_config=config,
        num_of_seeds=3,
        continuous=True,
    )
    collect_set_of_experiments('Pendulum-v0/initial_test')

