import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class critic_bilinear(nn.Module):
    def __init__(self, env_params):
        self.env_params = env_params
        super(critic_bilinear, self).__init__()
        self.max_action = env_params['action_max']
        # f(s, a)
        self.fc1_1 = nn.Linear(env_params['obs'] + env_params['action'], 176)
        self.fc1_2 = nn.Linear(176, 176)
        self.fc1_3 = nn.Linear(176, 176)
        self.fc1_4 = nn.Linear(176, 16)
        # phi(s, g)
        self.fc2_1 = nn.Linear(env_params['obs'] + env_params['goal'], 176)
        self.fc2_2 = nn.Linear(176, 176)
        self.fc2_3 = nn.Linear(176, 176)
        self.fc2_4 = nn.Linear(176, 16)

    def forward(self, x, actions):
        # f(s, a)
        x1 = torch.cat([x[...,:self.env_params['obs']], actions / self.max_action], dim=1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc1_2(x1))
        x1 = F.relu(self.fc1_3(x1))
        x1 = self.fc1_4(x1)
        # phi(s, g)
        x2 = F.relu(self.fc2_1(x))
        x2 = F.relu(self.fc2_2(x2))
        x2 = F.relu(self.fc2_3(x2))
        x2 = self.fc2_4(x2)
        #dot product
        q_value = torch.einsum('bs,bs->b', x1, x2)

        return q_value
