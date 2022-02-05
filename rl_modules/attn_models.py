import torch
import torch.nn as nn
import numpy as np

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
    dk = k.shape[-1]
    scaled_qk = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_qk += (mask * -1e9)  # 1: we don't want it, 0: we want it
    attention_weights = nn.functional.softmax(scaled_qk, dim=-1)  # (..., seq_len_q, seq_len_k)
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, feature_dim)
    return output, attention_weights

class SelfAttentionBase(nn.Module):
    def __init__(self, input_dim, feature_dim, n_heads=1):
        super(SelfAttentionBase, self).__init__()
        self.n_heads = n_heads
        self.q_linear = nn.Linear(input_dim, feature_dim)
        self.k_linear = nn.Linear(input_dim, feature_dim)
        self.v_linear = nn.Linear(input_dim, feature_dim)
        self.dense = nn.Linear(feature_dim, feature_dim)

    def split_head(self, x):
        x_size = x.size()
        assert isinstance(x_size[2] // self.n_heads, int)
        x = torch.reshape(x, [-1, x_size[1], self.n_heads, x_size[2] // self.n_heads])
        x = torch.transpose(x, 1, 2)  # (batch_size, n_heads, seq_len, depth)
        return x

    def forward(self, q, k, v, mask):
        assert len(q.size()) == 3
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q_heads = self.split_head(q)
        k_heads = self.split_head(k)
        v_heads = self.split_head((v))
        # mask = torch.unsqueeze(mask, dim=1).unsqueeze(dim=2)  # (batch_size, 1, 1, seq_len)
        attention_out, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        attention_out = torch.transpose(attention_out, 1, 2)  # (batch_size, seq_len_q, n_heads, depth)
        out_size = attention_out.size()
        attention_out = torch.reshape(attention_out, [-1, out_size[1], out_size[2] * out_size[3]])
        attention_out = self.dense(attention_out)
        return attention_out

class SelfAttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size, n_attention_blocks, n_heads):
        super(SelfAttentionExtractor, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(robot_dim + object_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )
        self.n_attention_blocks = n_attention_blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionBase(hidden_size, hidden_size, n_heads) for _ in range(n_attention_blocks)]
        )
        self.layer_norm1 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)])
        self.feed_forward_network = nn.ModuleList(
            [nn.ModuleList([nn.Linear(hidden_size, hidden_size),
                            nn.Linear(hidden_size, hidden_size)]) for _ in range(n_attention_blocks)])
        self.feed_forward_network = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
            ) for _ in range(n_attention_blocks)
        )
        self.layer_norm2 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)])   
    
    def forward(self, x):
        features = self.embed(x)
        for i in range(self.n_attention_blocks):
            attn_output = self.attention_blocks[i](features, features, features, mask = None)
            out1 = self.layer_norm1[i](features + attn_output)
            ffn_out = self.feed_forward_network[i](out1)
            features = self.layer_norm2[i](ffn_out)
        features = torch.mean(features, dim=1)
        return features
        
class AttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size, n_attention_blocks, n_heads):
        super(AttentionExtractor, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(robot_dim + object_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )
        self.n_attention_blocks = n_attention_blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionBase(hidden_size, hidden_size, n_heads) for _ in range(n_attention_blocks)]
        )
        self.layer_norm1 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)])
        self.feed_forward_network = nn.ModuleList(
            [nn.ModuleList([nn.Linear(hidden_size, hidden_size),
                            nn.Linear(hidden_size, hidden_size)]) for _ in range(n_attention_blocks)])
        self.feed_forward_network = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
            ) for _ in range(n_attention_blocks)
        )
        self.layer_norm2 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)])   
    
    def forward(self, x):
        features = self.embed(x)
        for i in range(self.n_attention_blocks):
            attn_output = self.attention_blocks[i](features, features, features, mask = None)
            out1 = self.layer_norm1[i](features + attn_output)
            ffn_out = self.feed_forward_network[i](out1)
            features = self.layer_norm2[i](ffn_out)
        features = torch.mean(features, dim=1)
        return features

class CrossAttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size):
        super(CrossAttentionExtractor, self).__init__()
        self.robot_dim = robot_dim
        self.object_dim = object_dim
        self.robot_embed = nn.Sequential(
            nn.Linear(robot_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
        )
        self.object_embed = nn.Sequential(
            nn.Linear(object_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU()
        )
        self.ln = nn.LayerNorm(hidden_size // 2)
    
    def forward(self, robot_obs, objects_obs):
        # robot_obs = x[:,0 , :self.robot_dim]
        # objects_obs = x[..., self.robot_dim:]
        robot_embedding = self.robot_embed(robot_obs)  # (batch_size, hidden_size/2)
        objects_embedding = self.object_embed(objects_obs)  # (batch_size, n_obj, hidden_size/2)
        weights = torch.matmul(robot_embedding.unsqueeze(dim=1), objects_embedding.transpose(1, 2)) / np.sqrt(objects_embedding.size()[2])  # (batch_size, 1, n_obj)
        weights = nn.functional.softmax(weights, dim=-1)  # (batch_size, 1, n_obj)
        weighted_feature = torch.matmul(weights, objects_embedding).squeeze(dim=1)  # (batch_size, hidden_size/2)
        weighted_feature = nn.functional.relu(self.ln(weighted_feature))
        return torch.cat([robot_embedding, weighted_feature], dim=-1)  # (batch_size, hidden_size)

class actor_attn(nn.Module):
    def __init__(self, env_params, cross=False):
        super(actor_attn, self).__init__()
        self.max_action = env_params['action_max']
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size']
        self.robot_obs_size = env_params['robot_obs_size']
        if cross:
            self.feature_extractor = CrossAttentionExtractor(self.robot_obs_size, self.obj_obs_size+self.goal_size, hidden_size = 64)
        else:
            self.feature_extractor = SelfAttentionExtractor(self.robot_obs_size, self.obj_obs_size+self.goal_size, hidden_size = 64, n_attention_blocks=4, n_heads=1)
        self.mlp = nn.Sequential(
            *(  [nn.Linear(64, 64), nn.ReLU()] +
                [nn.Linear(64, 64), nn.ReLU()] +
                [nn.Linear(64, env_params['action']), nn.Tanh()])
        )
        self._initialize()

    def forward(self, x):
        # robot_obs, objects_obs, masks = self.parse_obs(obs)
        x = self.preprocess(x)
        features = self.feature_extractor(x)
        actions = self.max_action * self.mlp(features)
        return actions

    def preprocess(self, x):
        '''
        obs: (batch_size, obs_size)
            obs_size = robot_obs_size + num_obj * object_obs_size + num_obj * goal_size
        return: (batch_size, num_obj, object_obs_size + robot_obs_size)
        '''
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
    
    def _initialize(self):
        for net in self.mlp:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=0.01)
                nn.init.constant_(net.bias, 0.)


class critic_attn(nn.Module):
    def __init__(self, env_params, cross=False):
        super(critic_attn, self).__init__()
        self.max_action = env_params['action_max']
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size']
        self.robot_obs_size = env_params['robot_obs_size']
        if cross:
            self.feature_extractor = CrossAttentionExtractor(self.robot_obs_size+env_params['action'], self.obj_obs_size+self.goal_size, hidden_size = 64)
        else:
            self.feature_extractor = SelfAttentionExtractor(self.robot_obs_size+env_params['action'], self.obj_obs_size+self.goal_size, hidden_size = 64, n_attention_blocks=4, n_heads=1)
        self.mlp = nn.Sequential(
            *(  [nn.Linear(64, 64), nn.ReLU()] +
                [nn.Linear(64, 64), nn.ReLU()] +
                [nn.Linear(64, 1)])
        )
        self._initialize()

    def forward(self, x, actions):
        # robot_obs, objects_obs, masks = self.parse_obs(obs)
        x = self.preprocess(x, actions)
        features = self.feature_extractor(x)
        q_value = self.mlp(features)
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
        robot_obs = x[:, :self.robot_obs_size].repeat(1,num_obj).reshape(batch_size, num_obj, self.robot_obs_size)
        act = act.repeat(1,num_obj).reshape(batch_size, num_obj, -1)
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        return torch.cat((act, robot_obs, obj_obs, goal_obs), dim=-1)

    def _initialize(self):
        for net in self.mlp:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)

class actor_crossattn(nn.Module):
    def __init__(self, env_params):
        super(actor_crossattn, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size']
        self.robot_obs_size = int(env_params['robot_obs_size'])
        assert self.robot_obs_size%2 == 0, f'robot obs {self.robot_obs_size} not %2=0'
        self.single_robot_obs_size = int(self.robot_obs_size/2)
        self.attn_1 = CrossAttentionExtractor(self.single_robot_obs_size, self.obj_obs_size+self.goal_size, hidden_size = 64)
        self.attn_2 = CrossAttentionExtractor(self.single_robot_obs_size, self.obj_obs_size+self.goal_size, hidden_size = 64)
        self.act_out = nn.Sequential(
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, env_params['action'])
        )

    def forward(self, x):
        batch_size, obs_size = x.shape
        # preprocess data
        assert (obs_size-self.robot_obs_size) % (self.obj_obs_size + self.goal_size) == 0, \
            f'Shape ERROR! obs_size{obs_size}, robot{self.robot_obs_size}, obj&goal{self.obj_obs_size+self.goal_size}'
        num_obj = int((obs_size-self.robot_obs_size) / (self.obj_obs_size + self.goal_size))
        robot_obs_1 = x[:, :self.single_robot_obs_size]
        robot_obs_2 = x[:, self.single_robot_obs_size:self.robot_obs_size]
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        # get action
        feature_1 = self.attn_1(robot_obs_1, torch.cat((obj_obs, goal_obs), dim=-1))
        feature_2 = self.attn_2(robot_obs_2, torch.cat((obj_obs, goal_obs), dim=-1))
        act = self.act_out(torch.cat((feature_1, feature_2), dim = -1))
        return act

class critic_crossattn(nn.Module):
    def __init__(self, env_params):
        super(critic_crossattn, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        self.goal_size = env_params['goal_size']
        self.obj_obs_size = env_params['obj_obs_size']
        self.robot_obs_size = int(env_params['robot_obs_size'])
        self.action_size = int(env_params['action'])
        assert self.robot_obs_size%2 == 0, f'robot obs {self.robot_obs_size} not %2=0'
        assert self.action_size%2 == 0, f'action_size {self.action_size} not %2=0'
        self.single_robot_obs_size = int(self.robot_obs_size/2)
        self.single_robot_act_size = int(self.action_size/2)
        self.attn_1 = CrossAttentionExtractor(self.single_robot_obs_size+self.single_robot_act_size, self.obj_obs_size+self.goal_size, hidden_size = 64)
        self.attn_2 = CrossAttentionExtractor(self.single_robot_obs_size+self.single_robot_act_size, self.obj_obs_size+self.goal_size, hidden_size = 64)
        self.q_out = nn.Sequential(
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, 128),nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, act):
        batch_size, obs_size = x.shape
        # preprocess data
        assert (obs_size-self.robot_obs_size) % (self.obj_obs_size + self.goal_size) == 0, \
            f'Shape ERROR! obs_size{obs_size}, robot{self.robot_obs_size}, obj&goal{self.obj_obs_size+self.goal_size}'
        num_obj = int((obs_size-self.robot_obs_size) / (self.obj_obs_size + self.goal_size))
        robot_obs_1 = x[:, :self.single_robot_obs_size]
        robot_act_1 = act[:, :self.single_robot_act_size]
        robot_obs_2 = x[:, self.single_robot_obs_size:self.robot_obs_size]
        robot_act_2 = act[:, self.single_robot_act_size:]
        obj_obs = x[:, self.robot_obs_size : self.robot_obs_size+self.obj_obs_size*num_obj]\
            .reshape(batch_size, num_obj, self.obj_obs_size)
        goal_obs = x[:, self.robot_obs_size+self.obj_obs_size*num_obj:]\
            .reshape(batch_size, num_obj, self.goal_size)
        # get q_value
        feature_1 = self.attn_1(torch.cat((robot_obs_1, robot_act_1), dim=-1), torch.cat((obj_obs, goal_obs), dim=-1))
        feature_2 = self.attn_2(torch.cat((robot_obs_2, robot_act_2), dim=-1), torch.cat((obj_obs, goal_obs), dim=-1))
        q_value = self.q_out(torch.cat((feature_1, feature_2), dim = -1))
        return q_value