from argparse import Action
import torch
import torch.nn as nn
import numpy as np

class critic_biattn(nn.Module):
    def __init__(self, env_params):
        super(critic_biattn, self).__init__()
        self.max_action = env_params['action_max']
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size']
        self.robot_obs_size = env_params['robot_obs_size']
        # f(s, a)
        self.f_in = nn.Sequential(
            nn.Linear(self.robot_obs_size + env_params['action'], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.f_out = nn.Sequential(
            nn.Linear(64, 176),
            nn.ReLU(),
            nn.Linear(176, 176),
            nn.ReLU(),
            nn.Linear(176, 16)
        )
        # phi(si, gi)
        self.phi_in = nn.Sequential(
            nn.Linear(self.obj_obs_size+self.goal_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.attn = nn.MultiheadAttention(embed_dim = 64, num_heads=1)
        self.phi_out = nn.Sequential(
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self._initialize()

    def forward(self, x, actions):
        # preprocess
        og, ra = self.preprocess(x, actions)
        # f(r, a)
        f_in = self.f_in(ra)
        f_out = self.f_out(f_in)
        # phi(o,g)
        phi_in = self.phi_in(og)
        attn, _ = self.attn(torch.movedim(f_in, 0, 1), \
            torch.movedim(phi_in, 0, 1), torch.movedim(phi_in, 0, 1))
        attn = torch.movedim(attn, 0, 1)
        phi_out = self.phi_out(attn)
        # dot product
        q_value = torch.einsum('bs,bs->b', f_out.squeeze(), phi_out.squeeze())
        return q_value

    def preprocess(self, x, act):
        '''
        obs: (batch_size, obs_size)
            obs_size = robot_obs_size + num_obj * object_obs_size + num_obj * goal_size
        return: (batch_size, num_obj, object_obs_size + robot_obs_size)
        '''
        batch_size, obs_size = x.shape
        assert (obs_size-self.robot_obs_size) % (self.obj_obs_size + self.goal_size) == 0
        num_obj = int((obs_size-self.robot_obs_size) / (self.obj_obs_size + self.goal_size))
        # obj state and goal sequence
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        og = torch.cat((obj_obs, goal_obs), dim = -1)
        # robot state and action sequence
        robot_obs = x[:, :self.robot_obs_size]
        ra = torch.cat((robot_obs, act), dim = -1).view(batch_size, 1, -1)
        return og, ra

    def _initialize(self):
        for net in zip(self.f_in, self.f_out, self.phi_in, self.phi_out):
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)
                
class actor_biattn(nn.Module):
    def __init__(self, env_params):
        super(actor_biattn, self).__init__()
        self.max_action = env_params['action_max']
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size']
        self.robot_obs_size = env_params['robot_obs_size']
        # f(s)
        self.f_in = nn.Sequential(
            nn.Linear(self.robot_obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # phi(si, gi)
        self.phi_in = nn.Sequential(
            nn.Linear(self.obj_obs_size+self.goal_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.attn = nn.MultiheadAttention(embed_dim = 64, num_heads=1)
        self.phi_out = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(64+16, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, env_params['action']), nn.Tanh()
        )
        self._initialize()

    def forward(self, x):
        # preprocess
        og, r = self.preprocess(x)
        # f(r)
        f_in = self.f_in(r)
        # phi(o,g)
        phi_in = self.phi_in(og)
        attn, _ = self.attn(torch.movedim(f_in, 0, 1), \
            torch.movedim(phi_in, 0, 1), torch.movedim(phi_in, 0, 1))
        attn = torch.movedim(attn, 0, 1)
        phi_out = self.phi_out(attn)
        # mlp
        features = torch.cat((f_in, phi_out), dim=-1)
        actions = self.max_action * self.mlp(features)
        return actions.squeeze()

    def preprocess(self, x):
        '''
        obs: (batch_size, obs_size)
            obs_size = robot_obs_size + num_obj * object_obs_size + num_obj * goal_size
        return: (batch_size, num_obj, object_obs_size + robot_obs_size)
        '''
        batch_size, obs_size = x.shape
        assert (obs_size-self.robot_obs_size) % (self.obj_obs_size + self.goal_size) == 0
        num_obj = int((obs_size-self.robot_obs_size) / (self.obj_obs_size + self.goal_size))
        # obj state and goal sequence
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        og = torch.cat((obj_obs, goal_obs), dim = -1)
        # robot state and action sequence
        robot_obs = x[:, :self.robot_obs_size]
        r = robot_obs.view(batch_size, 1, -1)
        return og, r

    def _initialize(self):
        for net in zip(self.f_in, self.phi_in, self.phi_out):
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)
        for net in self.mlp:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=0.01)
                nn.init.constant_(net.bias, 0.)


class critic_biselfattn(nn.Module):
    def __init__(self, env_params):
        super(critic_biselfattn, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size']
        self.robot_obs_size = env_params['robot_obs_size']
        # f(s, a)
        self.f = nn.Sequential(
            nn.Linear(env_params['obs'] + env_params['action'], 176), 
            nn.ReLU(),
            nn.Linear(176, 176),
            nn.ReLU(),
            nn.Linear(176, 176),
            nn.ReLU(),
            nn.Linear(176, 16)
        )
        # phi(si, gi)
        self.phi_in = nn.Sequential(
            nn.Linear(self.robot_obs_size+self.obj_obs_size+self.goal_size, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.attn = nn.MultiheadAttention(embed_dim = 64, num_heads=1)
        self.phi_out = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

    def forward(self, x, actions):
        batch_size, obs_size = x.shape
        # f(s, a)
        sa = torch.cat([x[...,:self.env_params['obs']], actions / self.max_action], dim=1)
        f = self.f(sa)
        # phi(s, g)
        assert (obs_size-self.robot_obs_size) % (self.obj_obs_size + self.goal_size) == 0, \
            f'Shape ERROR! obs_size{obs_size}, robot{self.robot_obs_size}, obj&goal{self.obj_obs_size+self.goal_size}'
        num_obj = int((obs_size-self.robot_obs_size) / (self.obj_obs_size + self.goal_size))
        robot_obs = x[:, :self.robot_obs_size].repeat(1,num_obj).reshape(batch_size, num_obj, self.robot_obs_size)
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        sg = torch.cat((robot_obs, obj_obs, goal_obs), dim=-1)
        phi_in = self.phi_in(sg)
        attn, _ = attn, _ = self.attn(torch.movedim(phi_in, 0, 1), \
            torch.movedim(phi_in, 0, 1), torch.movedim(phi_in, 0, 1))
        attn = torch.movedim(attn, 0, 1).reshape(batch_size, 64*num_obj)
        phi = self.phi_out(attn)
        # dot product
        q_value = torch.einsum('bs,bs->b', f, phi).reshape(-1, 1)

        return q_value