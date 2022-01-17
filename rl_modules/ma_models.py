from numpy import double
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
        super(actor_separated, self).__init__()
        self.max_action = env_params['action_max']
        self.num_agents = env_params['num_agents']
        self.partial_obs_size = int(env_params['obs']/self.num_agents)
        self.partial_action_size = int(env_params['action']/self.num_agents)
        self.goal_size = env_params['goal']
        self.module_list = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.partial_obs_size + self.goal_size, 128), 
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.partial_action_size),
                nn.Tanh()
            )] * self.num_agents)

    def forward(self, x):
        batch_size, obs_size = x.shape
        all_obs = x[..., :-self.goal_size].reshape(batch_size, self.num_agents, self.partial_obs_size)
        goal = x[..., -self.goal_size:].repeat(1, self.num_agents).reshape(batch_size, self.num_agents, self.goal_size)
        x = torch.cat((all_obs, goal), dim = -1)
        act = torch.Tensor()
        for i, module in enumerate(self.module_list):
            act = torch.cat((act, self.max_action*module(x[:, i, :])), dim = 1)
        return act.reshape(batch_size, self.num_agents*self.partial_action_size)

class actor_dropout(nn.Module):
    def __init__(self, env_params):
        super(actor_dropout, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        self.drop_out_rate = env_params['drop_out_rate']
        self.num_agents = env_params['num_agents']
        self.partial_obs_size = int(env_params['obs']/self.num_agents)
        self.partial_action_size = int(env_params['action']/self.num_agents)
        self.goal_size = env_params['goal']

    def forward(self, x):
        batch_size, obs_size = x.shape
        goal = x[..., -self.goal_size:].repeat(1, self.num_agents)\
            .reshape(batch_size, self.num_agents, self.goal_size)
        obs = x[..., :-self.goal_size].repeat(1, self.num_agents)\
            .reshape(batch_size, self.num_agents, self.partial_obs_size*self.num_agents)
        mat = torch.tensor([1]*self.partial_obs_size)
        full_mask = torch.block_diag(*[mat]*self.num_agents)\
            .reshape(1,self.num_agents,self.partial_obs_size*self.num_agents)\
            .repeat(batch_size,1,1)
        mask_coef = (torch.rand((batch_size,self.num_agents))<self.drop_out_rate)\
            .reshape(batch_size, self.num_agents, 1).repeat(1,1,self.partial_obs_size*self.num_agents)
        mask = full_mask * mask_coef + torch.ones_like(full_mask) * torch.logical_not(mask_coef)
        x = torch.cat((obs*mask, goal), dim = -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        mat = torch.tensor([1]*self.partial_action_size)
        act_mask = torch.block_diag(*[mat]*self.num_agents)\
            .reshape(1,self.num_agents,self.partial_action_size*self.num_agents)\
            .repeat(batch_size,1,1)
        actions = (act_mask*actions).sum(dim=1)

        return actions

class actor_multihead(nn.Module):
    def __init__(self, env_params):
        super(actor_multihead, self).__init__()
        self.max_action = env_params['action_max']
        self.num_agents = env_params['num_agents']
        self.partial_obs_size = int(env_params['obs']/self.num_agents)
        self.partial_action_size = int(env_params['action']/self.num_agents)
        self.goal_size = env_params['goal']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal']*self.num_agents, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        batch_size, obs_size = x.shape
        goal = x[..., -self.goal_size:].repeat(1, self.num_agents)\
            .reshape(batch_size, self.num_agents, self.goal_size)
        obs = x[..., :-self.goal_size]\
            .reshape(batch_size, self.num_agents, self.partial_obs_size)
        og = torch.cat((goal, obs), dim=-1).reshape(batch_size, -1).repeat(1, self.num_agents)\
            .reshape(batch_size, self.num_agents, self.num_agents*(self.partial_obs_size+self.goal_size))
        mat = torch.tensor([1]*(self.partial_obs_size+self.goal_size))
        full_mask = torch.block_diag(*[mat]*self.num_agents)\
            .reshape(1,self.num_agents,(self.partial_obs_size+self.goal_size)*self.num_agents)\
            .repeat(batch_size,1,1)
        x = og*full_mask
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        mat = torch.tensor([1]*self.partial_action_size)
        act_mask = torch.block_diag(*[mat]*self.num_agents)\
            .reshape(1,self.num_agents,self.partial_action_size*self.num_agents)\
            .repeat(batch_size,1,1)
        actions = (act_mask*actions).sum(dim=1)

        return actions