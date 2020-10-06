import os
from copy import deepcopy

import numpy
from muzero_general.muzero import MuZero

from muzero_general_experiments.config import MuZeroDefaultConfig
from muzero_general_experiments.games import MuZeroCartPoleGame, MuZeroPendulumGame


def main(num_seeds):
    r = numpy.random
    r.seed(0)
    seeds = r.randint(0, 1e5, size=num_seeds)

    cart_pole_game_cls = MuZeroCartPoleGame
    pendulum_game_cls = MuZeroPendulumGame

    default_config_cartpole = MuZeroDefaultConfig()
    default_config_cartpole.set_tag('default_cartpole')
    default_config_pendulum = MuZeroDefaultConfig()
    default_config_pendulum.support_size = 45
    default_config_pendulum.observation_shape = (1, 1, 3)
    default_config_pendulum.set_tag('default_pendulum')

    q_learning_comparison_config_cartpole = deepcopy(default_config_cartpole)
    q_learning_comparison_config_cartpole.PER = False  # Turn Off Prioritized Replay
    q_learning_comparison_config_cartpole.use_last_model_value = False  # Turn Off 'Reanalyze'
    q_learning_comparison_config_cartpole.set_tag('q_learning_cartpole')

    q_learning_comparison_config_pendulum = deepcopy(default_config_pendulum)
    q_learning_comparison_config_pendulum.PER = False  # Turn Off Prioritized Replay
    q_learning_comparison_config_pendulum.use_last_model_value = False  # Turn Off 'Reanalyze'
    q_learning_comparison_config_pendulum.set_tag('q_learning_pendulum')

    experiments = [
        [cart_pole_game_cls, default_config_cartpole],
        [cart_pole_game_cls, q_learning_comparison_config_cartpole],
        [pendulum_game_cls, default_config_pendulum],
        [pendulum_game_cls, q_learning_comparison_config_pendulum]
    ]

    for seed in seeds:
        for [game_cls, config] in experiments:
            config_updated_seed = deepcopy(config)
            config_updated_seed.seed = int(seed)
            config_updated_seed.set_tag(config_updated_seed.tag + '_' + str(seed))
            # if not os.path.exists(config_updated_seed.result_path):
            muzero = MuZero(game_cls_and_config=[game_cls, config_updated_seed])
            muzero.train(log_in_tensorboard=True)


if __name__ == '__main__':
    main(3)
