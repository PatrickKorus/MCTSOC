import ipympl
import matplotlib.pyplot as plt
import gym
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
import glob, os

from utils.memory import Memory
from utils.model import DeterministicModel
from utils.sampler import Sampler

def tuple_to_torch(tuple, requires_grad=False):
    torch_tensor = torch.from_numpy(np.stack(tuple)).type(torch.FloatTensor)
    if requires_grad:
        torch_tensor = Variable(torch_tensor, requires_grad=True)
    return torch_tensor

env = gym.make('MountainCar-v0')
run_tag = "reimplemented_00"
env.seed(1); torch.manual_seed(1); np.random.seed(1)
PATH = "tboardlogs/"#glob.glob(os.path.expanduser('~/tboardlogs/'))[0]
writer = SummaryWriter('tboardlogs/{}_{}'.format(datetime.now().strftime('%b%d_%H-%M-%S'), run_tag))

# Initialize Policy
policy = DeterministicModel(
    in_features=env.observation_space.shape[0],
    num_actions=env.action_space.n,
)
learning_rate = 0.01
num_epochs = 2000
epsilon = 0.3
batch_size = 2
train_batch_size = 1200
gamma = 0.99
loss_fn = nn.MSELoss()
optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
sampler = Sampler(env=env)
optimizer_steps = 3

memory = Memory(max_size=1200)

for episode in trange(num_epochs):

    # epsilon greedy policy
    def control(state, epsilon_greedy=True):
        Q = policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))

        if epsilon_greedy and np.random.rand(1) < epsilon:
            action = np.random.randint(0, 3)
        else:
            _, action = torch.max(Q, dim=-1)
            action = action.item()

        return action

    # sample
    new_data, info = sampler.get_batch(batch_size=batch_size,
                                       control=control,
                                       render_first=False,
                                       clipping=False,
                                       clipping_factor=1.0,
                                       return_info=True)
    writer.add_scalar('data/sum_of_rewards', info["sum_of_rewards"].mean(), episode)

    max_pos = (max(np.stack(new_data.sample().state)[:, 0]))
    writer.add_scalar('data/max_position', max_pos, episode)
    memory.append(new_data)
    writer.add_scalar('debug/memory_size', len(memory), episode)
    # train
    batch = memory.sample(batch_size=train_batch_size)
    states = tuple_to_torch(batch.state)
    actions = tuple_to_torch(batch.action)
    next_states = tuple_to_torch(batch.next_state)
    rewards = tuple_to_torch(batch.reward)

    for it in range(optimizer_steps):
        q = policy(Variable(states))
        q_target = q.clone().data
        q1 = policy(Variable(next_states))
        q1_max, _ = torch.max(q1, -1)
        for i in range(min(train_batch_size, len(q_target))):
            q_target[i, int(actions[i])] = rewards[i] + torch.mul(q1_max[i].detach(), gamma)
        loss = loss_fn(q, q_target)
        writer.add_scalar("data/optimizer_loss", loss, episode * optimizer_steps + it)
        policy.zero_grad()
        q = policy(states)
        loss.backward()
        optimizer.step()

writer.close()


X = np.random.uniform(-1.2, 0.6, 10000)
Y = np.random.uniform(-0.07, 0.07, 10000)
Z = []
for i in range(len(X)):
    _, temp = torch.max(policy(Variable(torch.from_numpy(np.array([X[i],Y[i]]))).type(torch.FloatTensor)), dim =-1)
    z = temp.item()
    Z.append(z)
Z = pd.Series(Z)
colors = {0:'blue',1:'lime',2:'red'}
colors = Z.apply(lambda x:colors[x])
labels = ['Left','Right','Nothing']

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
fig = plt.figure(3, figsize=[7,7])
ax = fig.gca()
plt.set_cmap('brg')
surf = ax.scatter(X, Y, c=Z)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_title('Policy')
recs = []
for i in range(0,3):
    try:
        recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    except IndexError as e:
        print(e)
        pass
plt.legend(recs,labels,loc=4,ncol=3)
fig.savefig('Policy.png')
plt.show()






