import gym
from mcts_general.common.wrapper import DiscreteActionWrapper, ManipulatedTimeDiscretization
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
    default_config.do_roll_outs = False

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
    default_config.do_roll_outs = False

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
    cartpole_discrete_from_continuous_more_actions = DiscreteGymGame(
        env=DiscreteActionWrapper(ContinuousCartPoleEnv(), num_actions=4, damping=1.0))
    pendulum_discrete_more_actions = DiscreteGymGame(
        env=DiscreteActionWrapper(gym.make('Pendulum-v0'), num_actions=4, damping=1.0))
    mountaincar_discrete_fewer_actions = DiscreteGymGame(
        env=DiscreteActionWrapper(gym.make('MountainCarContinuous-v0'))
    )

    all_experiments.append(MCTSExperiment('discrete_from_continuous', cartpole_discrete_from_continuous, default_config))
    all_experiments.append(MCTSExperiment('more_actions', pendulum_discrete_more_actions, default_config))
    all_experiments.append(MCTSExperiment('more_actions', cartpole_discrete_from_continuous_more_actions, default_config))
    all_experiments.append(MCTSExperiment('fewer_actions', mountaincar_discrete_fewer_actions, default_config))

    # time discretization in planning
    for n in [2, 4, 8]:
        tag = 'time_step_skipping_x{}'.format(n)

        cartpole_discrete_default_n_time = GymGameDoingMultipleStepsInSimulations(
            env=gym.make('CartPole-v0'),
            number_of_multiple_actions_in_simulation=n)  # v1 just has longer periods and higher reward threshold

        pendulum_discrete_default_n_time = GymGameDoingMultipleStepsInSimulations(
            env=DiscreteActionWrapper(
                gym.make('Pendulum-v0'),
                num_actions=2,
                damping=1.0),
            number_of_multiple_actions_in_simulation=2)

        mountaincar_discrete_default_n_time = GymGameDoingMultipleStepsInSimulations(
            env=gym.make('MountainCar-v0'),
            number_of_multiple_actions_in_simulation=n)

        all_experiments.append(MCTSExperiment(tag, cartpole_discrete_default_n_time, default_config))
        all_experiments.append(MCTSExperiment(tag, pendulum_discrete_default_n_time, default_config))
        all_experiments.append(MCTSExperiment(tag, mountaincar_discrete_default_n_time, default_config))

    # time discretization in planning and acting
    for n in [2, 4, 8]:
        tag = 'time_step_skipping_x{}_also_in_acting'.format(n)

        cartpole_discrete_default_n_time_in_acting = DiscreteGymGame(
            env=ManipulatedTimeDiscretization(
                gym.make('CartPole-v0'),
                number_of_time_steps_to_scip=n)
        )  # v1 just has longer periods and higher reward threshold

        pendulum_discrete_default_n_time_in_acting = DiscreteGymGame(
            env=ManipulatedTimeDiscretization(
                DiscreteActionWrapper(
                    gym.make('Pendulum-v0'),
                    num_actions=2,
                    damping=1.0),
                number_of_time_steps_to_scip=n)
        )

        mountaincar_discrete_default_n_time_in_acting = DiscreteGymGame(
            env=ManipulatedTimeDiscretization(
                gym.make('MountainCar-v0'),
                number_of_time_steps_to_scip=n)
        )

        all_experiments.append(MCTSExperiment(tag, cartpole_discrete_default_n_time_in_acting, default_config))
        all_experiments.append(MCTSExperiment(tag, pendulum_discrete_default_n_time_in_acting, default_config))
        all_experiments.append(MCTSExperiment(tag, mountaincar_discrete_default_n_time_in_acting, default_config))

    # engineered macro actions
    # pendulum_with_engineered_actions = PendulumGameWithEngineeredMacroActions(num_actions=2, action_damping=1.0)
    # all_experiments.append(MCTSExperiment('engineered_actions', pendulum_with_engineered_actions, default_config))

    # Variation of Theta and Alpha in Continuous Version
    for c in [1, 2, 3, 4]:
        for alpha in [0.0625, 0.125, 0.25, 0.5]:
            for sigma in [0.5, 0.75, 1.0]:
                spw_config = MCTSContinuousAgentConfig()
                spw_config.do_roll_outs = False
                spw_config.alpha = alpha
                spw_config.C = c
                tag = "SPW_alpha_{:1.3f}_C_{}_Sigma_{}".format(alpha, c, sigma)
                cartpole_continuous_default.sigma = sigma
                all_experiments.append(MCTSExperiment(tag, cartpole_continuous_default, spw_config))

                # Sigma * 2 for pendulum since action range is [-2, 2]
                tag = "SPW_alpha_{:1.3f}_C_{}_Sigma_{}".format(alpha, c, sigma * 2)
                pendulum_continuous_default.sigma = sigma * 2
                all_experiments.append(MCTSExperiment(tag, pendulum_continuous_default, spw_config))

    # Mountaincar
    config = MCTSAgentConfig()
    config.do_roll_outs = True
    config.do_roll_out_steps_with_simulation_true = True
    config.number_of_roll_outs = 10
    config.max_roll_out_depth = 30
    MCTSExperiment('deep_rollouts', mountaincar_discrete_default_n_time, config)

    return all_experiments


if __name__ == '__main__':
    all_exp = get_all_experiments()

    # mountaincar_discrete_default_n_time = GymGameDoingMultipleStepsInSimulations(
    #     env=gym.make('MountainCar-v0'),
    #     number_of_multiple_actions_in_simulation=4)
    # mountaincar_continuous_default = ContinuousGymGame(
    #     env=gym.make('MountainCarContinuous-v0'), mu=0., sigma=1.)
    # mountaincar_discrete_default_n_time = DiscreteGymGame(
    #     env=ManipulatedTimeDiscretization(gym.make('MountainCar-v0'), number_of_time_steps_to_scip=4)
    # )
    # config = MCTSAgentConfig()
    # config.do_roll_outs = True
    #
    # exp = MCTSExperiment('test', mountaincar_discrete_default_n_time, config)
    # exp.num_simulations = 800
    # exp.run(render=True)
    # pass