import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


def run_baseline_experiment(num_seeds=1):
    rand = np.random
    rand.seed(0)
    seeds = rand.randint(0, 1e5, size=num_seeds)
    env = gym.make('CartPole-v0')

    dqn = DQN(policy=MlpPolicy,
              env=env,
              learning_rate=5e-4,
              buffer_size=50000,
              learning_starts=1000,
              # tau: float = 1.0, # hard update
              train_freq=1,
              # gradient_steps: int = 1,
              # n_episodes_rollout: int = -1,
              target_update_interval=500,
              exploration_fraction=0.1,
              exploration_initial_eps=1.0,
              exploration_final_eps=0.02,
              # max_grad_norm: float = 10,
              # tensorboard_log: Optional[str] = None,
              # create_eval_env: bool = False,
              # policy_kwargs: Optional[Dict[str, Any]] = None,
              verbose=2,
              seed=0,
              # device: Union[th.device, str] = 'auto',
              # _init_setup_model: bool = True
              )

    # dqn.learn(total_timesteps=25000, log_interval=50)
    dqn.load("deepq_v3beta")
    obs = env.reset()

    while (True):
        action, state = dqn.predict(observation=obs, deterministic=True)
        env.render()
        obs, reward, done, _ = env.step(action)
        if done:
            env.close()
            break


if __name__ == '__main__':
    run_baseline_experiment(1)