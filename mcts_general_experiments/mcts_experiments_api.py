from multiprocessing import Pool

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

    @property
    def seed(self):
        return self.game.get_seed()

    @seed.setter
    def seed(self, seed):
        self.game.set_seed(seed)

    @property
    def num_simulations(self):
        return self.agent.mcts.config.num_simulations

    @num_simulations.setter
    def num_simulations(self, num_sim):
        self.agent.mcts.config.num_simulations = num_sim

    def get_copy(self):
        return MCTSExperiment(self.tag, self.game.get_copy(), deepcopy(self.agent.config))


def get_full_experiments_queue(
        experiments: typing.List[MCTSExperiment],
        num_of_seeds: int = 3,
        num_simulations_list: typing.List[int] = [10, 50, 100, 200, 400, 800, 1600, 3200]
):
    rand = np.random
    rand.seed(0)
    seeds = rand.randint(0, 1e5, size=num_of_seeds)

    experiment_queue = []
    for seed in seeds:
        for num_sim in num_simulations_list:
            for experiment in experiments:
                exp = experiment.get_copy()
                exp.seed = int(seed)
                exp.num_simulations = num_sim
                experiment_queue.append(exp)

    return experiment_queue


def run_experiment_and_store_results(experiment):
    try:
        path = '{}/{}'.format(experiment.game, experiment.tag)
        ensure_path(path)
        dump('{}/config.dump'.format(path), experiment.agent.config)

        print('starting experiment {} on {} with {} simulations and seed {}.'.format(
            experiment.tag,
            experiment.game,
            experiment.num_simulations,
            experiment.seed))
        tic = datetime.now()
        dump_target = '{}/{}_{}.dump'.format(path, experiment.seed, experiment.num_simulations)
        if os.path.isfile(dump_target):
            print("{} exists. Skipping...".format(dump_target))
            return

        result = experiment.run()
        result['seed'] = experiment.seed
        result['num_simulations'] = experiment.num_simulations
        result.to_pickle(dump_target)
        toc = datetime.now()
        print('Done experiment {} on {} with {} simulations and seed {}. Time:'.format(
            experiment.tag,
            experiment.game,
            experiment.num_simulations,
            experiment.seed,
            toc - tic))
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
