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
    games_classes = [cart_pole_game_cls, pendulum_game_cls]

    default_config = MuZeroDefaultConfig()

    q_learning_comparison_config = MuZeroDefaultConfig()
    q_learning_comparison_config.PER = False  # Turn Off Prioritized Replay
    q_learning_comparison_config.use_last_model_value = False  # Turn Off 'Reanalyze'

    configs = [default_config, q_learning_comparison_config]

    for seed in seeds:
        for game_cls in games_classes:
            for config in configs:
                config_updated_seed = deepcopy(config)
                config_updated_seed.seed = int(seed)
                muzero = MuZero(game_cls_and_config=[game_cls, config_updated_seed])
                muzero.train(log_in_tensorboard=True)


if __name__ == '__main__':
    main(3)
