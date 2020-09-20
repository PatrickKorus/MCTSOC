from mcts_general_experiments.all_experiments import all_experiments
from mcts_general_experiments.mcts_experiments_api import run_set_of_experiments

all_experiments = all_experiments

run_set_of_experiments(
    num_of_seeds=3,
    experiments=all_experiments,
)
