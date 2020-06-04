import time

#from quanser_robots import GentlyTerminating
from gym import Env
from utils.memory import Memory
import numpy as np


class Sampler:
    """
    A light weight sampler that saves env data to a memory object
    """
    def __init__(self, env: Env):
        """
        Initialize Sampler
        :param env: gym / quanser environment to sample data from
        """
        self.env = env

    def sample(self,
               number_of_samples,
               control=None,
               render=False,
               manual_done=False,
               end_with_done=True,
               render_slow_down_ms=0,
               clipping=True,
               clipping_factor=1.0):
        """
        Takes Control Function control(action) = state and tries to sample number_of_samples
        :param number_of_samples: number of samples to be taken from environment
        :param control: policy to be sampled from
        :param render: whether the sampling should be rendered
        :param manual_done: force the environment to continue sampling after "done" flag
        :param end_with_done: end the sampling, if the environment gets the done flag, if this is false, the environment
        will reset and sample again until it has number_of_samples
        :param render_slow_down_ms: if the rendering goes too fast, to be useful this parameter sets a delay in every
        render frame (in ms) to slow down the rendered animation
        :param clipping: clip the action to be between clipping_factor * action_space.low
        and clipping_factor * action_space.high
        :param clipping_factor: see clipping
        :return: Memory object with samples
        """

        memory = Memory()

        if not control:
            def control(state):
                return self.env.action_space.sample()

        state = self.env.reset()
        for it in range(number_of_samples):

            action_from_ctrl = control(state)

            if render:
                try:
                    self.env.render()
                    # this is to slow down rendering because of the rendered
                    # graphics being useless when GPU is too fast.
                    time.sleep(render_slow_down_ms * 1e-3)
                except NotImplementedError:
                    print("Rendering not possible for " + str(self.env) + "!")
                    render = False

            action = action_from_ctrl   # .copy()
            if clipping:
                np.clip(action,
                        clipping_factor * self.env.action_space.low,
                        clipping_factor * self.env.action_space.high,
                        action)

            next_state, reward, done, _ = self.env.step(action)
            if next_state[0] > -0.2:
                reward = 1
            if manual_done:
                # forces to continue staying in the roll out, has different effects depending on env
                done = it == number_of_samples - 1

            # this is to ensure consistency among the obs output of the environments
            next_state = np.array(next_state).flatten()
            memory.push(state, action_from_ctrl, done, next_state, reward)

            if done:
                if end_with_done:
                    break
                else:
                    state = self.env.reset()
            else:
                state = next_state
        self.env.close()
        return memory

    def get_batch(self,
                  batch_size,
                  max_trajectory_length=10000,
                  control=None,
                  render_first=False,
                  render_all=False,
                  render_slow_down_ms=0,
                  manual_done=False,
                  return_info=False,
                  clipping=True,
                  clipping_factor=1.0):
        """
        Takes Control Function control(action) = state and tries to sample number_of_samples
        :param batch_size: number of trajectories sampled
        :param max_trajectory_length:
        :param control: policy to be sampled from
        :param render_first: whether the first sample trajectory should be rendered
        :param render_all: whether the all sample trajectories should be rendered
        :param manual_done: force the environment to continue sampling after "done" flag
        :param render_slow_down_ms: if the rendering goes too fast, to be useful this parameter sets a delay in every
        render frame (in ms) to slow down the rendered animation
        :param clipping: clip the action to be between clipping_factor * action_space.low
        and clipping_factor * action_space.high
        :param clipping_factor: see clipping
        :return: Memory object with samples
        """
        memory = Memory()
        info = {"sum_of_rewards": np.zeros(batch_size)}

        for it in range(batch_size):

            render_flag = (it == 0 and render_first) or render_all

            trajectory = self.sample(max_trajectory_length,
                                     control,
                                     render=render_flag,
                                     render_slow_down_ms=render_slow_down_ms,
                                     manual_done=manual_done,
                                     clipping=clipping,
                                     clipping_factor=clipping_factor)

            # collect info
            if return_info:
                info["sum_of_rewards"][it] = np.stack(trajectory.sample().reward).sum()

            memory.append(trajectory)

        if return_info:
            return memory, info

        print(info["sum_of_rewards"][-1])
        return memory
