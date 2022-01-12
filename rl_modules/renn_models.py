import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, embedding_dim = 64, num_heads=1, softmax_temperature=1.0):
        super().__init__()
        self.fc_createheads = nn.Linear(embedding_dim, num_heads * embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(num_heads * embedding_dim, embedding_dim)
        self.softmax_temperature = nn.Parameter(torch.tensor(softmax_temperature))
        self.activation_fnx = F.leaky_relu

    def forward(self, query, context, memory):
        batch_size, num_obj, obs_size = query.size()
        num_head = int(self.fc_createheads.out_features / obs_size)
        query = self.fc_createheads(query).view(batch_size, num_obj, num_head, obs_size)
        query = query.unsqueeze(2).expand(-1, -1, memory.size(1), -1, -1) # TODO not understand
        context = context.unsqueeze(1).unsqueeze(3).expand_as(query)
        qc_logits = self.fc_logit(torch.tanh(context + query))
        attention_probs = F.softmax(qc_logits / self.softmax_temperature, dim=2)
        memory = memory.unsqueeze(1).unsqueeze(3).expand(-1, num_obj, -1, num_head, -1)
        attention_heads = (memory * attention_probs).sum(2).squeeze(2)
        attention_heads = self.activation_fnx(attention_heads)
        attention_result = self.fc_reduceheads(attention_heads.view(batch_size, num_obj, num_head*obs_size))
        return attention_result

class AttentiveGraphToGraph(nn.Module):
    """
    Uses attention to perform message passing between 1-hop neighbors in a fully-connected graph
    Change log
    1. remove mask
    """
    def __init__(self, embedding_dim=64, num_heads=1, layer_norm=True):
        super().__init__()
        self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.attention = Attention(embedding_dim, num_heads=num_heads)
        # self.layer_norm= nn.LayerNorm(3*embedding_dim) if layer_norm else None

    def forward(self, vertices):
        qcm_block = self.fc_qcm(vertices)
        query, context, memory = qcm_block.chunk(3, dim=-1)
        return self.attention(query, context, memory)

class AttentiveGraphPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 64
        self.init_w=3e-3
        self.num_heads=1
        self.input_independent_query = nn.Parameter(torch.Tensor(self.embedding_dim))
        self.input_independent_query.data.uniform_(-self.init_w, self.init_w)
        self.attention = Attention(embedding_dim = self.embedding_dim, num_heads=self.num_heads)

    def forward(self, vertices):
        batch_size, num_obj, obs_size = vertices.size()
        query = self.input_independent_query.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        context = vertices
        memory = vertices
        attention_result = self.attention(query, context, memory)
        return attention_result

class GraphPropagation(nn.Module):
    def __init__(self):
        super(GraphPropagation, self).__init__()
        self.num_query_heads = 1
        self.num_relational_blocks = 3
        self.embedding_dim = 64
        self.activation_fnx = F.leaky_relu
        self.graph_module_list = nn.ModuleList(
            [AttentiveGraphToGraph() for _ in range(self.num_relational_blocks)])
        # self.layer_norms = nn.ModuleList(
        #     [nn.LayerNorm(self.embedding_dim) for i in range(self.num_relational_blocks)])

    def forward(self, vertices):
        output = vertices
        for i in range(self.num_relational_blocks):
            new_output = self.graph_module_list[i](output)
            new_output = output + new_output
            output = self.activation_fnx(new_output)
            # output = self.layer_norms[i](output)
        return output

class actor_ReNN(nn.Module):
    def __init__(self, env_params):
        super().__init__()
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size'] #12 #15
        self.robot_obs_size = env_params['robot_obs_size'] #14 #10
        self.ignore_goal_size = env_params['ignore_goal_size'] #0 #3 # ignore gripper pos
        self.mlp_in = nn.Sequential(
            nn.Linear(self.robot_obs_size+self.obj_obs_size+self.goal_size, 64),
            # nn.LayerNorm(64)
        )
        self.graph_propagation = GraphPropagation()
        self.read_out = AttentiveGraphPooling()
        self.mlp_out = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, env_params['action'])
        )

    def forward(self, obs):
        obs = self.preprocess(obs)
        # ideal_vertices = torch.load('/Users/reedpan/Downloads/vertices')
        vertices = self.mlp_in(obs)
        # print(vertices==ideal_vertices)
        # ideal_embeddings = torch.load('/Users/reedpan/Downloads/embeddings')
        embeddings = self.graph_propagation(vertices)
        # ideal_selected_objects = torch.load('/Users/reedpan/Downloads/selected_objects')
        selected_objects = self.read_out(vertices=embeddings)
        selected_objects = selected_objects.squeeze(1)
        action = self.mlp_out(selected_objects)
        return action
    
    def preprocess(self, x):
        '''
        obs: (batch_size, obs_size)
            obs_size = robot_obs_size + num_obj * object_obs_size + num_obj * goal_size
        return: (batch_size, num_obj, object_obs_size + robot_obs_size)
        '''
        if self.ignore_goal_size > 0:
            x = x[...,:-self.ignore_goal_size] # ignore useless part
        batch_size, obs_size = x.shape
        assert (obs_size-self.robot_obs_size) % (self.obj_obs_size + self.goal_size) == 0, \
            f'Shape ERROR! obs_size{obs_size}, robot{self.robot_obs_size}, obj&goal{self.obj_obs_size+self.goal_size}'
        num_obj = int((obs_size-self.robot_obs_size) / (self.obj_obs_size + self.goal_size))
        robot_obs = x[:, :self.robot_obs_size].repeat(1,num_obj).reshape(batch_size, num_obj, self.robot_obs_size)
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        return torch.cat((robot_obs, obj_obs, goal_obs), dim=-1)

class critic_ReNN(nn.Module):
    def __init__(self, env_params):
        super().__init__()
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size'] #12 #15
        self.robot_obs_size = env_params['robot_obs_size'] #14 #10
        self.ignore_goal_size = env_params['ignore_goal_size'] #0 #3 # ignore gripper pos
        self.mlp_in = nn.Sequential(
            nn.Linear(env_params['action'] + self.robot_obs_size+self.obj_obs_size+self.goal_size, 64),
            # nn.LayerNorm(64)
        )
        self.graph_propagation = GraphPropagation()
        self.read_out = AttentiveGraphPooling()
        self.mlp_out = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act):
        obs = self.preprocess(obs, act)
        # ideal_vertices = torch.load('/Users/reedpan/Downloads/vertices')
        vertices = self.mlp_in(obs)
        # print(vertices==ideal_vertices)
        # ideal_embeddings = torch.load('/Users/reedpan/Downloads/embeddings')
        embeddings = self.graph_propagation(vertices)
        # ideal_selected_objects = torch.load('/Users/reedpan/Downloads/selected_objects')
        selected_objects = self.read_out(vertices=embeddings)
        selected_objects = selected_objects.squeeze(1)
        action = self.mlp_out(selected_objects)
        return action
    
    def preprocess(self, x, act):
        '''
        obs: (batch_size, obs_size)
            obs_size = robot_obs_size + num_obj * object_obs_size + num_obj * goal_size
        return: (batch_size, num_obj, object_obs_size + robot_obs_size)
        '''
        if self.ignore_goal_size > 0:
            x = x[...,:-self.ignore_goal_size] # ignore useless part
        batch_size, obs_size = x.shape
        assert (obs_size-self.robot_obs_size) % (self.obj_obs_size + self.goal_size) == 0
        num_obj = int((obs_size-self.robot_obs_size) / (self.obj_obs_size + self.goal_size))
        robot_obs = x[:, :self.robot_obs_size].repeat(1,num_obj).reshape(batch_size, num_obj, self.robot_obs_size)
        act = act.repeat(1,num_obj).reshape(batch_size, num_obj, -1)
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        return torch.cat((act, robot_obs, obj_obs, goal_obs), dim=-1)


if __name__ == '__main__':
    import gym
    import fetch_block_construction
    env = gym.make('FetchBlockConstruction_1Blocks_IncrementalReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1')
    obs = env.reset()
    env_param = {'obs': obs['observation'].shape[0],
        'goal': obs['desired_goal'].shape[0],
        'action': env.action_space.shape[0],
        'action_max': env.action_space.high[0],
    }
    actor = actor_ReNN(env_param)
    critic = critic_ReNN(env_param)
    # obs = torch.load('/Users/reedpan/Downloads/obs')
    for i in range(100):
        obs = torch.Tensor(np.append(obs['observation'], obs['desired_goal']).reshape(1, -1))
        action = (actor(obs)).detach().numpy().flatten()
        obs, rew, done, _ =  env.step(action)
        print(
            critic(
                torch.Tensor(np.append(obs['observation'], obs['desired_goal'])).reshape(1, -1), 
                torch.Tensor(action))
        )
        env.render(mode = 'human')