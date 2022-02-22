from ast import Not
from matplotlib.pyplot import axes, axis
import torch
import os
from datetime import datetime
from time import time
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, actor_bilinear, critic, critic_bilinear, critic_sum,\
    actor_large, critic_large
from rl_modules.renn_models import actor_ReNN, critic_ReNN
from rl_modules.attn_models import actor_attn, critic_attn, actor_crossattn, critic_crossattn
from rl_modules.biattn_models import critic_biattn, actor_biattn, critic_biselfattn
from rl_modules.ma_models import actor_shared, actor_separated, actor_dropout, actor_multihead,\
    actor_master_slave, actor_master, actor_attn_master_slave
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import wandb
from tqdm import tqdm

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        if 'use_task_distribution' in (self.args.env_kwargs.keys()):
            self.other_side_rate = 0.6
        # MPI
        self.comm = MPI.COMM_WORLD
        self.nprocs = self.comm.Get_size()
        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name, self.args.name)
        # create the network and target network
        if args.actor_shared:
            self.actor_network = actor_shared(env_params)
            self.actor_target_network = actor_shared(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        elif args.actor_separated:
            self.actor_network = actor_separated(env_params)
            self.actor_target_network = actor_separated(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        elif args.actor_master_slave:
            if self.args.use_attn:
                self.actor_network = actor_attn_master_slave(env_params, args.use_cross, args.num_blocks, master_only = args.master_only, shared_policy=self.args.shared_policy)
                self.actor_target_network = actor_attn_master_slave(env_params, args.use_cross, args.num_blocks, master_only = args.master_only, shared_policy=self.args.shared_policy)
                self.critic_network = critic_attn(env_params, args.use_cross, args.num_blocks)
                self.critic_target_network = critic_attn(env_params, args.use_cross, args.num_blocks)
            else:
                self.actor_network = actor_master_slave(env_params, self.env.obs_parser)
                self.actor_target_network = actor_master_slave(env_params, self.env.obs_parser)
                self.critic_network = critic(env_params)
                self.critic_target_network = critic(env_params)
        elif args.actor_master:
            self.actor_network = actor_master(env_params)
            self.actor_target_network = actor_master(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        elif args.actor_dropout:
            self.actor_network = actor_dropout(env_params)
            self.actor_target_network = actor_dropout(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        elif args.actor_multihead:
            self.actor_network = actor_multihead(env_params)
            self.actor_target_network = actor_multihead(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        elif args.use_renn:
            self.actor_network = actor_ReNN(env_params)
            self.actor_target_network = actor_ReNN(env_params)
            self.critic_network = critic_ReNN(env_params)
            self.critic_target_network = critic_ReNN(env_params)
        elif args.use_bilinear:
            self.actor_network = actor(env_params)
            self.actor_target_network = actor(env_params)
            self.critic_network = critic_bilinear(env_params)
            self.critic_target_network = critic_bilinear(env_params)
        elif args.use_critic_sum:
            self.actor_network = actor(env_params)
            self.actor_target_network = actor(env_params)
            self.critic_network = critic_sum(env_params)
            self.critic_target_network = critic_sum(env_params)
        elif args.use_attn:
            self.actor_network = actor_attn(env_params, args.use_cross, args.num_blocks)
            self.actor_target_network = actor_attn(env_params, args.use_cross, args.num_blocks)
            self.critic_network = critic_attn(env_params, args.use_cross, args.num_blocks)
            self.critic_target_network = critic_attn(env_params, args.use_cross, args.num_blocks)
        elif args.use_biattn:
            self.actor_network = actor_attn(env_params)
            self.actor_target_network = actor_attn(env_params)
            self.critic_network = critic_biattn(env_params)
            self.critic_target_network = critic_biattn(env_params)
        elif args.use_crossattn:
            self.actor_network = actor_crossattn(env_params)
            self.actor_target_network = actor_crossattn(env_params)
            self.critic_network = critic_crossattn(env_params)
            self.critic_target_network = critic_crossattn(env_params)
        elif args.actor_large:
            self.actor_network = actor_large(env_params)
            self.actor_target_network = actor_large(env_params)
            self.critic_network = critic_large(env_params)
            self.critic_target_network = critic_large(env_params)
        else:
            self.actor_network = actor(env_params)
            self.actor_target_network = actor(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        if self.args.learn_from_expert:
            assert args.resume, 'expert need model!'
            self.new_actor_loss  = []
            self.expert_network = actor(env_params).eval()
        # load paramters
        if args.resume:
            if self.args.model_path == None:
                path = os.path.join(self.args.save_dir, self.args.env_name, self.args.name, 'latest_model.pt')
            else:
                path = self.args.model_path
            o_dict, g_dict, actor_model, critic_model = torch.load(path, map_location=lambda storage, loc: storage)
            # OLD Version 
            # o_mean, o_std, g_mean, g_std, actor_model, critic_model = torch.load(path, map_location=lambda storage, loc: storage)
            print('loaded done!')
            if self.args.learn_from_expert:
                self.expert_network.load_state_dict(actor_model)
            elif self.args.actor_master_slave:
                self.actor_network.master_net.load_state_dict(actor_model)
                self.actor_network.slave_net.load_state_dict(actor_model)
            else:
                self.actor_network.load_state_dict(actor_model)
            self.critic_network.load_state_dict(critic_model)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward, random_unmoved = self.args.random_unmoved, not_relabel_unmoved = self.args.not_relabel_unmoved)
        # goal sampler
        # self.goal_sampler = goal_sampler()
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        if args.resume:
            # Note: if use object number curriculum, the normalizer need to be extended
            self.o_norm.load(o_dict)
            self.g_norm.load(g_dict)
            # OLD VERSION 
            # self.o_norm.mean = o_mean
            # self.o_norm.std = o_std
            # self.g_norm.mean = g_mean
            # self.g_norm.std = g_std
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            # if not os.path.exists(self.args.save_dir):
            #     os.mkdir(self.args.save_dir, exist_ok=True)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # start wandb to log
            if self.args.wandb:
                wandb.init(
                    project = self.args.project,
                    group = self.args.group,
                    tags = self.args.tags, 
                    name = self.args.name,
                    notes = f'Env:{self.args.env_name},Note:{self.args.note}'
                )

    def learn(self):
        """
        train the network

        """
        # warm up
        if self.args.warmup:
            self.warmup(100)
        # start to collect samples
        start_time = time()
        collect_per_epoch = self.args.n_cycles * self.args.num_rollouts_per_mpi * self.env_params['max_timesteps']
        self.global_relabel_rate = 0.3
        curriculum_param = self.args.curriculum_init
        curri_indicator = 0
        best_success_rate = 0
        total_steps = 0
        for epoch in range(self.args.n_epochs):
            # change task distribution
            if 'use_task_distribution' in (self.args.env_kwargs.keys()):
                self.env.task.other_side_rate = self.other_side_rate
                print('current os rate:', self.other_side_rate)
            # start curriculum
            if self.args.curriculum and curri_indicator > self.args.curriculum_bar:
                if curriculum_param < self.args.curriculum_end:
                    best_success_rate = 0
                    curriculum_param += self.args.curriculum_step
                    path = self.model_path + f'/curr{curriculum_param:.2f}_model.pt'
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        torch.save([self.o_norm.state_dict(), self.g_norm.state_dict(), self.actor_network.state_dict(), \
                            self.critic_network.state_dict()], path)
                        if self.args.wandb:
                            wandb.save(path)
                        print(f'save curriculum {curriculum_param:.2f} end model at {self.model_path}')
                if self.args.curriculum_type == 'env_param':
                    self.env.change(curriculum_param)
                elif self.args.curriculum_type == 'dropout':
                    self.actor_network.dropout_vel_rate = curriculum_param
                    self.actor_target_network.dropout_vel_rate = curriculum_param
                else:
                    raise NotImplementedError
                observation = self.env.reset()
                # extend normalizer to new observation
                o_size = len(observation['observation'])
                g_size = len(observation['desired_goal'])
                self.o_norm.change_size(new_size = o_size)
                self.g_norm.change_size(new_size = g_size)
                # extend buffer to new observation
                self.buffer.change_size(max_timesteps=self.env._max_episode_steps,\
                    obs_size=o_size, goal_size=g_size)
            num_useless_rollout = 0 # record number of useless rollout(ag not change)
            for _ in tqdm(range(self.args.n_cycles)):
                mb_obs, mb_ag, mb_g, mb_info, mb_actions = [], [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # try until collect successful experience
                    for j in range(self.args.max_trail_time):
                        # reset the rollouts
                        ep_obs, ep_ag, ep_g, ep_info, ep_actions = [], [], [], [], []
                        # reset the environment
                        observation = self.env.reset()
                        obs = observation['observation']
                        ag = observation['achieved_goal']
                        g = observation['desired_goal']
                        info = observation.get('info') # if no info, return None
                        # start to collect samples
                        ag_origin = ag
                        extra_reset_steps = np.random.randint(self.env._max_episode_steps) if self.args.extra_reset_steps and np.random.uniform() < 0.5 else 0
                        delay_agent = np.random.uniform() < 0.5
                        for t in range(self.env._max_episode_steps+extra_reset_steps):
                            with torch.no_grad():
                                input_tensor = self._preproc_inputs(obs, g)
                                if self.args.collect_from_expert:
                                    pi = self.expert_network(input_tensor)
                                elif self.args.actor_master_slave and not self.args.master_only:
                                    input_tensor_mirror = self._preproc_inputs(obs, g, mirror=True)
                                    pi = self.actor_network(input_tensor, input_tensor_mirror)
                                else:
                                    pi = self.actor_network(input_tensor)
                                action = self._select_actions(pi)
                            # if in extra_reset_steps steps, only step one action
                            if t < extra_reset_steps:
                                ag_origin = ag
                                if delay_agent:
                                    action = np.append(action[:4], np.array([0,0,0,-1]))
                                else:
                                    action = np.append(np.array([0,0,0,-1]), action[4:])
                            # feed the actions into the environment
                            observation_new, _, _, info = self.env.step(action)
                            total_steps += 1
                            # self.env.render()
                            obs_new = observation_new['observation']
                            ag_new = observation_new['achieved_goal']
                            # append rollouts
                            if t >= extra_reset_steps:
                                ep_obs.append(obs.copy())
                                ep_ag.append(ag.copy())
                                ep_g.append(g.copy())
                                ep_info.append(info.copy())
                                ep_actions.append(action.copy())
                            # re-assign the observation
                            obs = obs_new
                            ag = ag_new
                        # check if use this rollout
                        if_moved = np.linalg.norm(ag.reshape(-1,self.args.dim) - ag_origin.reshape(-1,self.args.dim), axis=-1) > 0.005
                        if_drop = np.any(ag.reshape(-1,self.args.dim)[..., -1] < (-0.1)) and self.args.ignore_drop
                        if self.args.trail_mode == 'all':
                            if_moved = if_moved.all()
                        elif self.args.trail_mode == 'any':
                            if_moved = if_moved.any()
                        else:
                            raise NotImplementedError
                        if if_moved and (not if_drop):
                            break
                        else:
                            num_useless_rollout += 1
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_info.append(ep_info)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_info = np.array(mb_info)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_info, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_info, mb_actions])
                # train the network
                self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            data = self._eval_agent(render = ((epoch%10)==0  and self.args.render))
            case_data = {}
            for k,v in self.args.eval_kwargs.items():
                if isinstance(v, list):
                    for j in v:
                        case_data[f'{k}{j:.2f}'] = self._eval_agent(render = ((epoch%10)==0  and self.args.render), eval_kwargs = {k:j})['success_rate']
                else:
                    case_data[k] = self._eval_agent(render = ((epoch%10)==0  and self.args.render), eval_kwargs = {k:v})['success_rate']
            rates = []
            for k,v in case_data.items():
                rates.append(1/(v+0.1))
            if len(case_data) > 0:
                self.other_side_rate = 1 - np.array(rates)[0]/sum(rates)
            curri_indicator = data[self.args.curriculum_indicator]
            # record relabel rate
            local_relabel_rate = self.her_module.relabel_num/self.her_module.total_sample_num
            local_random_relabel_rate = self.her_module.random_num/self.her_module.total_sample_num
            local_not_relabel_rate = self.her_module.nochange_num/self.her_module.total_sample_num
            local_data = np.array([local_relabel_rate, local_random_relabel_rate, local_not_relabel_rate])
            global_data = np.zeros(3)
            self.comm.Allreduce(local_data, global_data, op=MPI.SUM)
            self.global_relabel_rate, global_random_relabel_rate, global_not_relabel_rate = global_data/self.nprocs
            # local
            if MPI.COMM_WORLD.Get_rank() == 0:
                # save data
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, reward is: {:.3f}'.format(datetime.now(), epoch, data['success_rate'], data['reward']))
                torch.save([self.o_norm.state_dict(), self.g_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/latest_model.pt')
                if self.args.wandb:
                    wandb.save(self.model_path + '/latest_model.pt')
                if data['success_rate'] > best_success_rate:
                    best_success_rate = data['success_rate']
                    torch.save([self.o_norm.state_dict(), self.g_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + f'/curr{curriculum_param:.2f}_best_model.pt')
                    if self.args.wandb:
                        wandb.save(self.model_path + f'/curr{curriculum_param:.2f}_best_model.pt')
                    print(f'save curriculum {curriculum_param:.2f} best model at {self.model_path}')
                if self.args.wandb:
                    # log data
                    log_data = {
                        'success rate': data['success_rate'], 
                        "reward": data['reward'], 
                        "curriculum param": curriculum_param, 
                        "run time": (time()-start_time)/3600, 
                        "useless rollout per epoch": num_useless_rollout/(self.args.n_cycles*self.args.num_rollouts_per_mpi),
                        "future relabel rate": self.global_relabel_rate, 
                        "random relabel rate": global_random_relabel_rate, 
                        "not change relabel rate": global_not_relabel_rate, 
                        **case_data
                    }
                    if hasattr(self, 'other_side_rate'):
                        log_data['other_side_rate'] = self.other_side_rate
                    wandb.log(
                        log_data, 
                        step=total_steps
                    )
            # reset record parameters
            self.her_module.total_sample_num = 1
            self.her_module.relabel_num = 0
            self.her_module.random_num = 0
            self.her_module.nochange_num = 0

    # pre_process the inputs
    def _preproc_inputs(self, obs, g, mirror = False, not_unsqueeze = False):
        if mirror:
            x = np.concatenate((obs, g), axis=-1)
            x = self.env.obs_parser(x, mode='mirror')
            obs = x[..., :obs.shape[-1]]
            g = x[..., obs.shape[-1]:]
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate((obs_norm, g_norm), axis=-1)
        if not_unsqueeze:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        else:
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_info, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'info': mb_info, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        if self.args.dynamic_batch: # update according to buffer size
            update_times = int(self.args.n_batches * self.buffer.current_size / self.buffer.size)
        elif self.args.her_batch:
            update_times = int(self.args.n_batches / self.global_relabel_rate)
        else:
            update_times = self.args.n_batches * int(self.env._max_episode_steps/50)
        
        for _ in range(update_times):
            # sample the episodes
            transitions = self.buffer.sample(self.args.batch_size)
            # pre-process the observation and goal
            o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
            transitions['obs'], transitions['g'] = self._preproc_og(o, g)
            transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
            # start to do the update
            obs_norm = self.o_norm.normalize(transitions['obs'])
            g_norm = self.g_norm.normalize(transitions['g'])
            inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
            obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
            g_next_norm = self.g_norm.normalize(transitions['g_next'])
            inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
            # transfer them into the tensor
            inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
            inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
            actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
            r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
            if self.args.cuda:
                inputs_norm_tensor = inputs_norm_tensor.cuda()
                inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
                actions_tensor = actions_tensor.cuda()
                r_tensor = r_tensor.cuda()
            # calculate the target Q value function
            with torch.no_grad():
                # do the normalization
                # concatenate the stuffs
                if self.args.actor_master_slave and not self.args.master_only:
                    inputs_next_norm_tensor_mirror = self._preproc_inputs(transitions['obs_next'], transitions['g_next'], mirror=True, not_unsqueeze=True)
                    actions_next = self.actor_target_network(inputs_next_norm_tensor, inputs_next_norm_tensor_mirror)
                else:
                    actions_next = self.actor_target_network(inputs_next_norm_tensor)
                q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                # clip the q value
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            # the q loss
            real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            # the actor loss
            if self.args.actor_master_slave and not self.args.master_only:
                inputs_norm_tensor_mirror = self._preproc_inputs(transitions['obs'], transitions['g'], mirror=True, not_unsqueeze=True)
                actions_real = self.actor_network(inputs_norm_tensor, inputs_norm_tensor_mirror)
            else:
                actions_real = self.actor_network(inputs_norm_tensor)
            q_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
            actor_loss = self.args.q_coef * q_loss
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
            if self.args.learn_from_expert:
                actions_expert = self.expert_network(inputs_norm_tensor)
                imitate_loss = (actions_real - actions_expert).pow(2).mean()
                actor_loss += self.args.imitate_coef * imitate_loss
            # start to update the network
            self.actor_optim.zero_grad()
            actor_loss.backward()
            sync_grads(self.actor_network)
            self.actor_optim.step()
            # update the critic_network
            self.critic_optim.zero_grad()
            critic_loss.backward()
            sync_grads(self.critic_network)
            self.critic_optim.step()
        

    # do the evaluation
    def _eval_agent(self, render = False, eval_new_actor = False, eval_kwargs = {}):
        total_success_rate = []
        total_reward = []
        # record video
        if MPI.COMM_WORLD.Get_rank() == 0:
            video = []
        for n in range(self.args.n_test_rollouts):
            per_success_rate = []
            per_reward = []
            observation = self.env.reset(**eval_kwargs)
            obs = observation['observation']
            g = observation['desired_goal']
            extra_reset_steps = np.random.randint(self.env._max_episode_steps) if self.args.extra_reset_steps and np.random.uniform()<0.5 else 0
            delay_agent = np.random.uniform()<0.5
            for t in range(self.env_params['max_timesteps']+extra_reset_steps):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    if self.args.learn_from_expert and eval_new_actor:
                        pi = self.new_actor_network(input_tensor)
                    elif self.args.actor_master_slave and not self.args.master_only:
                        input_tensor_mirror = self._preproc_inputs(obs, g, mirror=True)
                        pi = self.actor_network(input_tensor, input_tensor_mirror)
                    else:
                        pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                if t < extra_reset_steps:
                    if delay_agent:
                        actions = np.append(actions[:4], np.array([0,0,0,-1]))
                    else:
                        actions = np.append(np.array([0,0,0,-1]), actions[4:])
                observation_new, reward, _, info = self.env.step(actions)
                # self.env.render(mode='human')
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if t >= extra_reset_steps:
                    per_success_rate.append(info['is_success'])
                    per_reward.append(reward)
                if MPI.COMM_WORLD.Get_rank() == 0 and render:
                    frame = np.array(self.env.render(mode = 'rgb_array'))
                    frame = np.moveaxis(frame, -1, 0)
                    video.append(frame)
            total_success_rate.append(per_success_rate)
            total_reward.append(per_reward)
        if MPI.COMM_WORLD.Get_rank() == 0 and render and self.args.wandb:
            wandb.log({f"video{eval_kwargs}": wandb.Video(np.array(video), fps=30, format="mp4")})
        total_success_rate = np.array(total_success_rate)
        total_reward = np.array(total_reward)
        local_success_rate = np.mean(total_success_rate[:, -1])
        local_reward = np.mean(total_reward[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_reward = MPI.COMM_WORLD.allreduce(local_reward, op=MPI.SUM)
        return {
            'success_rate': global_success_rate / MPI.COMM_WORLD.Get_size(),
            'reward': global_reward / MPI.COMM_WORLD.Get_size(),
        }
    
    def warmup(self, num_rollout, policy = 'random', store = True):
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(num_rollout):
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environmentz
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            for t in range(self.env_params['max_timesteps']):
                if policy == 'random':
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs, g)
                        if self.args.actor_master_slave and not self.args.master_only:
                            input_tensor_mirror = self._preproc_inputs(obs, g, mirror=True)
                            pi = self.actor_network(input_tensor, input_tensor_mirror)
                        else:
                            pi = self.actor_network(input_tensor)
                        action = self._select_actions(pi)
                # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            if store: # record final obs if store
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        # store the episodes
        if store:
            self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            self._update_normalizer([mb_obs, mb_ag, mb_g , mb_actions])
        else:
            obs_norm = self.o_norm.normalize(mb_obs)
            g_norm = self.g_norm.normalize(mb_g)
            inputs_norm = np.concatenate([obs_norm, g_norm], axis=-1)
            assert inputs_norm.shape == (num_rollout, self.env_params['max_timesteps'],self.env_params['obs']+self.env_params['goal']), inputs_norm.shape
            obs_tensor = torch.tensor(inputs_norm, dtype=torch.float32).cuda()\
                .reshape(num_rollout*self.env_params['max_timesteps'], self.env_params['goal']+self.env_params['obs'])
            act_tensor = torch.tensor(mb_actions, dtype=torch.float32).cuda()\
                .reshape(num_rollout*self.env_params['max_timesteps'], self.env_params['action'])
            return obs_tensor, act_tensor

    def _update_new_actor(self):
        # evaluate old policy
        data = self._eval_agent()
        assert data['success_rate'] > 0.9, 'success rate:'+data['success_rate']+' is too low!'
        # use supervised learning to update new policy
        self.actor_network.eval() # fix old actor
        self.new_actor_network.cuda()
        # collect enough data to train new policy
        for _ in tqdm(range(10)):
            obs_tensor, act_tensor = self.warmup(10000, policy = 'old_policy', store = False)
            num_epoch = 20
            num_batch = 100
            data_size = obs_tensor.shape(0)
            batch_size = int(data_size/num_batch)
            success_rate = 0
            for epoch in range(num_epoch):
                for i in range(num_batch):
                    self.new_actor_optim.zero_grad()
                    actions_new = self.new_actor_network(obs_tensor[batch_size*i:batch_size*(i+1)])
                    new_actor_loss = (actions_new - act_tensor[batch_size*i:batch_size*(i+1)]).pow(2).mean()
                    new_actor_loss.backward()
                    self.new_actor_optim.step()
                data = self._eval_agent(eval_new_actor=True)
                print('epoch:', epoch, ' success rate: ', data['success_rate'])
                if data['success_rate'] > 0 and (data['success_rate'] - success_rate)<0.0001:
                    break
                success_rate = data['success_rate']
        #     if self.args.wandb and MPI.COMM_WORLD.Get_rank() == 0:
        #         self.new_actor_loss.append(new_actor_loss.detach().cpu())
        #         if len(self.new_actor_loss) == 2000:
        #             wandb.log({"actor loss": np.mean(self.new_actor_loss)})
        #             self.new_actor_loss = []
        # self.actor_network.train()
        exit()