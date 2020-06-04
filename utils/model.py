"""
Taken from here https://github.com/transedward/pytorch-dqn/blob/master/dqn_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicModel(nn.Module):
    def __init__(self, in_features=4, num_actions=18, hidden_nodes_l1=64,  hidden_nodes_l2=64):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_nodes_l1, bias=True)
        self.fc2 = nn.Linear(hidden_nodes_l1, hidden_nodes_l2, bias=True)
        self.out = nn.Linear(hidden_nodes_l2, num_actions, bias=False)
        self.out.weight.data.mul_(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

    def forward_numpy(self, x, dtype=torch.float64):
        """ For processing numpy input """
        with torch.no_grad():
            x = torch.tensor(x, dtype=dtype).unsqueeze(0)
            output = self.forward(x)
        return output.squeeze(0).data.numpy()
