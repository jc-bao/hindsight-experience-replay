from unittest.mock import NonCallableMagicMock
from numpy import double
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn_models import actor_attn

"""
Multi-agent Modules

"""

# define the actor network
class actor_shared(nn.Module):
    def __init__(self, env_params, identification = True):
        # Note: id for agent is important
        super(actor_shared, self).__init__()
        self.identification = identification
        self.max_action = env_params['action_max']
        self.num_agents = env_params['num_agents']
        self.partial_obs_size = int(env_params['obs']/self.num_agents)
        self.partial_action_size = int(env_params['action']/self.num_agents)
        self.goal_size = env_params['goal']
        input_size = self.partial_obs_size + env_params['goal']
        if self.identification: input_size+=1
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, self.partial_action_size)

    def forward(self, x):
        batch_size, obs_size = x.shape
        all_obs = x[..., :-self.goal_size].reshape(batch_size, self.num_agents, self.partial_obs_size)
        goal = x[..., -self.goal_size:].repeat(1, self.num_agents).reshape(batch_size, self.num_agents, self.goal_size)
        x = torch.cat((all_obs, goal), dim = -1)
        if self.identification:
            i = torch.arange(-1, 1, 2/self.num_agents).view(1, self.num_agents, 1).repeat(batch_size, 1, 1)
            x = torch.cat((i, x), dim = -1)
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

# class actor_master_slave(nn.Module):
#     def __init__(self, env_params):
#         super(actor_master_slave, self).__init__()
#         self.max_action = env_params['action_max']
#         self.num_agents = env_params['num_agents']
#         self.partial_action_size = int(env_params['action']/self.num_agents)
#         self.goal_size = env_params['goal']
#         self.master_module = nn.Sequential(
#             nn.Linear(env_params['obs'] + self.partial_action_size + self.goal_size, 176), 
#             nn.ReLU(),
#             nn.Linear(176, 176),
#             nn.ReLU(),
#             nn.Linear(176, 176),
#             nn.ReLU(),
#             nn.Linear(176, self.partial_action_size),
#             nn.Tanh()
#         )
#         self.slave_module = nn.Sequential(
#             nn.Linear(env_params['obs'] + self.goal_size, 176), 
#             nn.ReLU(),
#             nn.Linear(176, 176),
#             nn.ReLU(),
#             nn.Linear(176, 176),
#             nn.ReLU(),
#             nn.Linear(176, self.partial_action_size),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         act_slave = self.slave_module(x)
#         act_master = self.master_module(torch.cat((x, act_slave), dim = -1))
#         return torch.cat((act_slave, act_master), dim = -1)

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


class actor_master(nn.Module):
    def __init__(self, env_params):
        super(actor_master, self).__init__()
        self.single_act_size = int(env_params['action']/2)
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, self.single_act_size+4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))[..., :self.single_act_size]
        actions = torch.cat((actions, torch.zeros_like(actions)), dim=-1)
        return actions

class actor_master_slave(nn.Module):
    def __init__(self, env_params):
        super(actor_master_slave, self).__init__()
        self.single_act_size = int(env_params['action']/2)
        self.max_action = env_params['action_max']
        # self.master_net = nn.Sequential(
        #     nn.Linear(env_params['obs'] + env_params['goal'], 256),nn.ReLU(),
        #     nn.Linear(256, 256),nn.ReLU(),
        #     nn.Linear(256, 256),nn.ReLU(),
        #     nn.Linear(256, self.single_act_size+4), nn.Tanh()
        # )
        self.master_net = actor_master(env_params)
        self.slave_net = actor_master(env_params)
        # self.master_fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        # self.master_fc2 = nn.Linear(256, 256)
        # self.master_fc3 = nn.Linear(256, 256)
        # self.master_action_out = nn.Linear(256, self.single_act_size+4)
        # self.slave_net = nn.Sequential(
        #     nn.Linear(env_params['obs'] + env_params['goal'], 256),nn.ReLU(),
        #     nn.Linear(256, 256),nn.ReLU(),
        #     nn.Linear(256, 256),nn.ReLU(),
        #     nn.Linear(256, self.single_act_size+4), nn.Tanh()
        # )
        # self.slave_fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        # self.slave_fc2 = nn.Linear(256, 256)
        # self.slave_fc3 = nn.Linear(256, 256)
        # self.slave_action_out = nn.Linear(256, self.single_act_size+4)

    def forward(self, x, x_mirror):

        # x=[robot obs, obj obs, goal]
        # master_act = self.master_net(x)[..., :self.single_act_size]*self.max_action
        # master_act=F.relu(self.fc1(x))
        # master_act = F.relu(self.fc2(master_act))
        # master_act = F.relu(self.fc3(master_act))
        # actions = self.max_action * torch.tanh(self.action_out(master_act))[..., :self.single_act_size]
        # slave_act = self.slave_net(self.obs_parser(x, mode='mirror'))[..., :self.single_act_size]*self.max_action
        # slave_act=F.relu(self.fc1(self.obs_parser(x)))
        # slave_act = F.relu(self.fc2(slave_act))
        # slave_act = F.relu(self.fc3(slave_act))
        # actions = self.max_action * torch.tanh(self.action_out(slave_act))[..., :self.single_act_size]
        # actions = torch.cat((self.master_net(x)[...,:self.single_act_size], self.slave_net(self.obs_parser(x,mode='mirror'))[...,:self.single_act_size]*torch.Tensor([-1,-1,1,1])), dim=-1)
        actions = torch.cat((self.master_net(x)[...,:self.single_act_size], self.slave_net(x_mirror)[...,:self.single_act_size]*torch.Tensor([-1,-1,1,1])), dim=-1)
        return actions

class actor_attn_master_slave(nn.Module):
    def __init__(self, env_params, cross=False, num_blocks=4, master_only = False, shared_policy = False):
        super(actor_attn_master_slave, self).__init__()
        self.single_act_size = int(env_params['action']/2)
        self.master_net = actor_attn(env_params, cross=cross, num_blocks=num_blocks)
        if not master_only:
            if shared_policy:
                self.slave_net = self.master_net
            else:
                self.slave_net = actor_attn(env_params, cross=cross, num_blocks=num_blocks)


    def forward(self, x, x_mirror=None):
        master_act = self.master_net(x)[...,:self.single_act_size]
        if not x_mirror == None:
            slave_act = self.slave_net(x_mirror)[...,:self.single_act_size]*torch.Tensor([-1,-1,1,1])
        else:
            slave_act = torch.zeros_like(master_act)
        actions = torch.cat((master_act, slave_act), dim=-1)
        return actions