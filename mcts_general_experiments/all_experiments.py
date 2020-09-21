import gym
from mcts_general.common.wrapper import DiscreteActionWrapper
from mcts_general.config import MCTSContinuousAgentConfig, MCTSAgentConfig
from mcts_general.game import DiscreteGymGame, ContinuousGymGame, GymGameDoingMultipleStepsInSimulations, \
    PendulumGameWithEngineeredMacroActions

from common.cartpole_continuous import ContinuousCartPoleEnv
from mcts_general_experiments.mcts_experiments_api import MCTSExperiment


def get_all_experiments():
    all_experiments = []

    # discrete games
    tag = "standard"

    default_config = MCTSAgentConfig()
    # default_config.do_roll_outs = False

    cartpole_discrete_default = DiscreteGymGame(
        env=gym.make('CartPole-v0'))  # v1 just has longer periods and higher reward threshold
    pendulum_discrete_default = DiscreteGymGame(
        env=DiscreteActionWrapper(gym.make('Pendulum-v0'), num_actions=2, damping=1.0))
    mountaincar_discrete_default = DiscreteGymGame(
        env=gym.make('MountainCar-v0'))

    all_experiments.append(MCTSExperiment(tag, cartpole_discrete_default, default_config))
    all_experiments.append(MCTSExperiment(tag, pendulum_discrete_default, default_config))
    all_experiments.append(MCTSExperiment(tag, mountaincar_discrete_default, default_config))

    # continuous games
    tag = "continuous"

    default_config = MCTSContinuousAgentConfig()
    # default_config.do_roll_outs = False

    cartpole_continuous_default = ContinuousGymGame(
        env=ContinuousCartPoleEnv(), mu=0.0, sigma=1.0)
    pendulum_continuous_default = ContinuousGymGame(
        env=gym.make('Pendulum-v0'), mu=0., sigma=2.)
    mountaincar_continuous_default = ContinuousGymGame(
        env=gym.make('MountainCarContinuous-v0'), mu=0., sigma=1.)

    all_experiments.append(MCTSExperiment(tag, cartpole_continuous_default, default_config))
    all_experiments.append(MCTSExperiment(tag, pendulum_continuous_default, default_config))
    all_experiments.append(MCTSExperiment(tag, mountaincar_continuous_default, default_config))


    # variants
    cartpole_discrete_from_continuous = DiscreteGymGame(
        env=DiscreteActionWrapper(ContinuousCartPoleEnv(), num_actions=2, damping=1.0))
    pendulum_discrete_more_actions = DiscreteGymGame(
        env=DiscreteActionWrapper(gym.make('Pendulum-v0'), num_actions=4, damping=1.0))
    mountaincar_discrete_fewer_actions = DiscreteGymGame(
        env=DiscreteActionWrapper(gym.make('MountainCarContinuous-v0'))
    )

    all_experiments.append(MCTSExperiment('discrete_from_continuous', cartpole_discrete_from_continuous, default_config))
    all_experiments.append(MCTSExperiment('more_actions', pendulum_discrete_more_actions, default_config))
    all_experiments.append(MCTSExperiment('fewer_actions', mountaincar_discrete_fewer_actions, default_config))

    # time discretization
    for n in [2, 4, 8]:
        tag = 'time_step_skipping_x{}'.format(n)

        cartpole_discrete_default_n_time = GymGameDoingMultipleStepsInSimulations(
            env=gym.make('CartPole-v0'), number_of_multiple_actions_in_simulation=n)  # v1 just has longer periods and higher reward threshold
        pendulum_discrete_default_n_time = GymGameDoingMultipleStepsInSimulations(
            env=DiscreteActionWrapper(gym.make('Pendulum-v0'), num_actions=n, damping=1.0), number_of_multiple_actions_in_simulation=2)
        mountaincar_discrete_default_n_time = GymGameDoingMultipleStepsInSimulations(
            env=gym.make('MountainCar-v0'), number_of_multiple_actions_in_simulation=n)

        all_experiments.append(MCTSExperiment(tag, cartpole_discrete_default_n_time, default_config))
        all_experiments.append(MCTSExperiment(tag, pendulum_discrete_default_n_time, default_config))
        all_experiments.append(MCTSExperiment(tag, mountaincar_discrete_default_n_time, default_config))

    # engineered macro actions
    pendulum_with_engineered_actions = PendulumGameWithEngineeredMacroActions(num_actions=2, action_damping=1.0)
    all_experiments.append(MCTSExperiment('engineered_actions', pendulum_with_engineered_actions, default_config))
    # TODO: variation of theta and alpha in SPW...
    return all_experiments
