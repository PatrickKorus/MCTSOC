import gym
import numpy as np, pandas as pd
from mcts_general.common.wrapper import DiscreteActionWrapper
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

from utils.basic_io import ensure_path


def run_baseline_experiment(num_seeds=1):
    rand = np.random
    rand.seed(0)
    seeds = rand.randint(0, 1e5, size=num_seeds)
    result = []
    envs = ['CartPole-v0', 'Pendulum-v0']
    for seed in seeds:
        # env = DiscreteActionWrapper(gym.make('Pendulum-v0'), num_actions=2, damping=1.)
        env = gym.make('CartPole-v0')
        env.seed(int(seed))


        # unfortunately the logging & evaluation of the model is not yet implemented in stable_baselines3
        # this is why we will do this slow workaround to get a decent
        for it in range(0, 20001, 200):
            dqn = DQN(policy=MlpPolicy,
                      env=env,
                      learning_rate=5e-4,
                      buffer_size=500 * 200,  # MuZero Buffer Size * Episode Length
                      learning_starts=100,
                      # tau: float = 1.0, # hard update
                      train_freq=1,
                      # gradient_steps: int = 1,
                      # n_episodes_rollout: int = -1,
                      target_update_interval=50,
                      exploration_fraction=0.1,
                      exploration_initial_eps=1.0,
                      exploration_final_eps=0.02,
                      # max_grad_norm: float = 10,
                      tensorboard_log=None,
                      # create_eval_env: bool = False,
                      # policy_kwargs: Optional[Dict[str, Any]] = None,
                      verbose=0,
                      seed=3,
                      # device: Union[th.device, str] = 'auto',
                      # _init_setup_model: bool = True,
                      )
            dqn.set_random_seed(int(seed))
            dqn.learn(total_timesteps=it,
                      log_interval=50)


            n_evaluation_rollouts = 10
            total_reward = 0
            for _ in range(n_evaluation_rollouts):
                # done = False
                obs = env.reset()
                while (True):
                    action, state = dqn.predict(observation=obs, deterministic=True)
                    # env.render()
                    obs, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        env.close()
                        break
            total_reward /= n_evaluation_rollouts
            result.append({'seed': seed, 'total reward': total_reward, 'env': str(env), 'total_time_steps': it})
            print({'seed': seed, 'total reward': total_reward, 'env': str(env), 'total_time_steps': it})
    df = pd.DataFrame(result)
    result_path = 'result_baseline/cartpole_tighter_updates.pkl'
    ensure_path(result_path)
    df.to_pickle(result_path)


if __name__ == '__main__':
    run_baseline_experiment(3)
