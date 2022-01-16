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

class actor_bilinear(nn.Module):
    def __init__(self, env_params):
        super(actor_bilinear, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size']
        self.robot_obs_size = env_params['robot_obs_size']
        # pi(s, g)
        self.pi = nn.Sequential(
            nn.Linear(env_params['obs'] + env_params['goal'], 176), 
            nn.ReLU(),
            nn.Linear(176, 176),
            nn.ReLU(),
            nn.Linear(176, 176),
            nn.ReLU(),
            nn.Linear(176, env_params['action'])
        )

    def forward(self, x):
        batch_size, obs_size = x.shape
        # pi(s, g)
        assert (obs_size-self.robot_obs_size) % (self.obj_obs_size + self.goal_size) == 0, \
            f'Shape ERROR! obs_size{obs_size}, robot{self.robot_obs_size}, obj&goal{self.obj_obs_size+self.goal_size}'
        num_obj = int((obs_size-self.robot_obs_size) / (self.obj_obs_size + self.goal_size))
        robot_obs = x[:, :self.robot_obs_size].repeat(1,num_obj).reshape(batch_size, num_obj, self.robot_obs_size)
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        sg = torch.cat((robot_obs, obj_obs, goal_obs), dim=-1)
        pi = self.pi(sg)
        pi = torch.sum(pi, axis=-2)

        return pi

class critic_bilinear(nn.Module):
    def __init__(self, env_params):
        self.env_params = env_params
        super(critic_bilinear, self).__init__()
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
        # phi(s, g)
        self.phi = nn.Sequential(
            nn.Linear(env_params['obs'] + env_params['goal'], 176), 
            nn.ReLU(),
            nn.Linear(176, 176),
            nn.ReLU(),
            nn.Linear(176, 176),
            nn.ReLU(),
            nn.Linear(176, 16)
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
        phi = self.phi(sg)
        phi = torch.sum(phi, axis=-2)
        #dot product
        q_value = torch.einsum('bs,bs->b', f, phi).reshape(-1, 1)

        return q_value

class critic_sum(nn.Module):
    def __init__(self, env_params):
        super(critic_sum, self).__init__()
        self.max_action = env_params['action_max']
        self.num_obj = int(env_params['goal']/3)
        self.fc1 = nn.Linear(env_params['obs'] + 3 + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.num_goal = 1

    def forward(self, x, actions):
        goal = x[..., -self.num_obj*3:]
        obs = x[..., :-self.num_obj*3]
        q_value = torch.zeros(x.size(dim=0), 1, dtype=torch.float32)
        for i in range(self.num_goal):
            x = torch.cat([obs, goal[...,i*3:i*3+3], actions / self.max_action], dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            q_value += self.q_out(x)

        return q_value

# define the actor network
class actor_large(nn.Module):
    def __init__(self, env_params):
        super(actor_large, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.action_out = nn.Linear(512, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class critic_large(nn.Module):
    def __init__(self, env_params):
        super(critic_large, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.q_out = nn.Linear(512, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value