import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Multi-agent Modules

"""

# define the actor network
class actor_shared(nn.Module):
    def __init__(self, env_params):
        super(actor_shared, self).__init__()
        self.max_action = env_params['action_max']
        self.num_agents = env_params['num_agents']
        self.partial_obs_size = int(env_params['obs']/self.num_agents)
        self.partial_action_size = int(env_params['action']/self.num_agents)
        self.goal_size = env_params['goal']
        self.fc1 = nn.Linear(self.partial_obs_size + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, self.partial_action_size)

    def forward(self, x):
        batch_size, obs_size = x.shape
        all_obs = x[..., :-self.goal_size].reshape(batch_size, self.num_agents, self.partial_obs_size)
        goal = x[..., -self.goal_size:].repeat(1, self.num_agents).reshape(batch_size, self.num_agents, self.goal_size)
        x = torch.cat((all_obs, goal), dim = -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions.reshape(batch_size, self.num_agents*self.partial_action_size)

class actor_separated(nn.Module):
    def __init__(self, env_params):
        super(actor_shared, self).__init__()
        self.max_action = env_params['action_max']
        self.num_agents = env_params['num_agents']
        self.partial_obs_size = int(env_params['obs']/self.num_agents)
        self.partial_action_size = int(env_params['action']/self.num_agents)
        self.goal_size = env_params['goal']
        self.module_list = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.partial_obs_size + self.goal_size, 64), 
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.partial_action_size),
                self.max_action*nn.Tanh()
            )] * self.num_agents)

    def forward(self, x):
        batch_size, obs_size = x.shape
        all_obs = x[..., :-self.goal_size].reshape(batch_size, self.num_agents, self.partial_obs_size)
        goal = x[..., -self.goal_size:].repeat(1, self.num_agents).reshape(batch_size, self.num_agents, self.goal_size)
        x = torch.cat((all_obs, goal), dim = -1)
        act = torch.Tensor()
        for i in self.num_agents:
            act = torch.cat((act, self.module_list[i](x[:, i, :])), dim = 1)
        return act.reshape(batch_size, self.num_agents*self.partial_action_size)