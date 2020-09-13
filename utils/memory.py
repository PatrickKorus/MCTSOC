from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state', 'reward'))


class Memory:
    """
    Class for handling Data. After some attempts by ourselves we figured this would be the best
    way to go as it is used in most gym examples of reinforcement learning algorithms we
    looked at.

    Taken from
    https://github.com/Khrylx/PyTorch-RL/blob/master/utils/replay_memory.py
    who has taken it from
    https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
    with some extensions for filtering.
    """

    def __init__(self, max_size=None):
        self.memory = []
        self.max_size = max_size

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        if self.max_size is not None and len(self) > self.max_size:
            self.memory = self.memory[-self.max_size:-1]

    def sample(self, batch_size=None):
        """Returns batch from samples"""
        if batch_size is None or batch_size >= len(self):
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        """Extends Memory by other Memory object"""
        self.memory += new_memory.memory
        if self.max_size is not None and len(self) > self.max_size:
            self.memory = self.memory[-self.max_size:-1]


    def __len__(self):
        return len(self.memory)


class MeanStdFilter:
    """
    For filtering/normalizing data as of x_norm = (x - mean) / std
    Using combined Bessel-Corrected std and combined mean explained here:
    https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
    For Bessel-Correction see:
    Weisstein, Eric W. "Bessel's Correction." From MathWorld--A Wolfram Web Resource.
    http://mathworld.wolfram.com/BesselsCorrection.html
    """

    def __init__(self, data=None):
        """
        :param data: initial data
        """
        self.mean = None
        self.std = None
        self.n = 0
        if data:
            self.push(data)

    def push(self, list_of_numpy_values):
        """
        Takes a list of numpy values to update mean and std.
        :param list_of_numpy_values: The batch to update std and mean upon
        """
        old_mean = self.mean
        old_std = self.std
        old_n = self.n
        new_mean = list_of_numpy_values.mean(axis=0)
        new_std = list_of_numpy_values.std(axis=0)
        new_n = len(list_of_numpy_values)

        if old_n == 0:
            self.mean = new_mean
            self.std = new_std
            self.n = new_n
        else:
            self.n = old_n + new_n
            self.mean = (old_n * old_mean + new_n * new_mean) / self.n
            # Combined standard deviation using "Bessel's Correction"
            self.std = np.sqrt(((old_n-1)*old_std**2 + (new_n-1)*new_std**2 + old_n*(old_mean-self.mean)**2
                                + new_n*(new_mean-self.mean)**2) / (self.n - 1))

    def normalize(self, numpy_stack):
        """
        Scales the arrays in the numpy stack in terms of x_dash = (x - mean) / std
        :param numpy_stack: an array of numpy arrays
        :return: the rescaled stack
        """
        return (numpy_stack - self.mean)/self.std if self.n > 0 and np.all(self.std > 1e-9) else numpy_stack

    def reproduce(self, numpy_stack):
        """
        Inverse of "normalize"
        :param numpy_stack:  an array of numpy arrays
        :return: the "unscaled" stack
        """
        return (numpy_stack * self.std) + self.mean if self.n > 0 and np.all(self.std > 1e-9) else numpy_stack
