import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


def run_baseline_experiment(num_seeds=1):
    rand = np.random
    rand.seed(0)
    seeds = rand.randint(0, 1e5, size=num_seeds)

    for seed in seeds:
        env = gym.make('CartPole-v0')

        dqn = DQN(policy=MlpPolicy,
                  env=env,
                  learning_rate=5e-4,
                  buffer_size=500*200,  # MuZero Buffer Size * Episode Length
                  learning_starts=1000,
                  # tau: float = 1.0, # hard update
                  train_freq=1,
                  # gradient_steps: int = 1,
                  # n_episodes_rollout: int = -1,
                  target_update_interval=500,
                  exploration_fraction=0.1,
                  exploration_initial_eps=1.0,
                  exploration_final_eps=0.05,
                  # max_grad_norm: float = 10,
                  tensorboard_log = 'result_q_learing',
                  # create_eval_env: bool = False,
                  # policy_kwargs: Optional[Dict[str, Any]] = None,
                  verbose=0,
                  seed=3,
                  # device: Union[th.device, str] = 'auto',
                  # _init_setup_model: bool = True,
                  )

        # unfortunately the logging & evaluation of the model is not yet implemented in stable_baselines3
        # this is why we will do this slow workaround to get a decent
        for it in range(10):
            dqn.learn(total_timesteps=20000,
                      log_interval=50,
                      eval_log_path='result_q_learning_log_path',
                      tb_log_name='result_q_learning_log_path_2')
        # dqn.save("deepq_v3beta")
            obs = env.reset()

            n_evaluation_rollouts = 3
            total_reward = 0
            for _ in range(n_evaluation_rollouts):
                while (True):
                    action, state = dqn.predict(observation=obs, deterministic=True)
                    # env.render()
                    obs, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        env.close()
                        break
            total_reward /= n_evaluation_rollouts
            print(total_reward)


if __name__ == '__main__':
    run_baseline_experiment(1)