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

    def __init__(self, tag: str, game: DeepCopyableGame, config: MCTSAgentConfig):
        self.game = game
        self.tag = tag

        if isinstance(game, ContinuousGymGame):
            assert isinstance(config, MCTSContinuousAgentConfig), \
                'Config must be suited for continuous game {}!'.format(game)
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

    def seed(self, seed):
        self.game.seed(seed)

    def num_simulations(self, num_simulations):
        self.agent.mcts.config.num_simulations = num_simulations


def run_set_of_experiments(
        num_of_seeds: int,
        experiments: typing.List[MCTSExperiment],
        num_simulations_list: typing.List[int] = [10, 50, 100, 200, 400, 800, 1600, 3200]
):
    rand = np.random
    rand.seed(0)
    seeds = rand.randint(0, 1e5, size=num_of_seeds)

    for seed in seeds:
        for num_sim in num_simulations_list:
            it = 0
            for experiment in experiments:
                it += 1
                try:
                    game = experiment.game
                    path = '{}/{}'.format(game, experiment.tag)
                    ensure_path(path)
                    if it == 1:
                        dump('{}/config.dump'.format(path), experiment.agent.config)

                    print('starting experiment {} with {} simulations and seed {}.'.format(experiment.tag, num_sim, seed))
                    tic = datetime.now()
                    experiment.seed(seed)
                    experiment.num_simulations(num_sim)
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
                except Exception as e:
                    warning('Experiment {} with {} failed due to {}!'.format(experiment, experiment.game, e))


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
    pass
