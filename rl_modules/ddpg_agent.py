from ast import Not
from matplotlib.pyplot import axes, axis
import torch
import os
from datetime import datetime
from time import time
import numpy as np
from mpi4py import MPI
from utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
import rl_modules.models as models
from utils.normalizer import normalizer
from her_modules.her import her_sampler
from her_modules.air import AIR
import wandb
from tqdm import tqdm

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, config, env, env_params):
        self.config = config
        self.env = env
        self.env_params = env_params
        self._max_episode_steps = env_params['max_episode_steps']
        # MPI
        self.comm = MPI.COMM_WORLD
        self.nprocs = self.comm.Get_size()
        # path to save the model
        self.model_path = os.path.join(self.config.save_dir, self.config.env_name, self.config.name)
        # create the network and target network
        actor_model = getattr(models, self.config.actor_model)
        critic_model = getattr(models, self.config.critic_model)
        self.actor_network = actor_model(env_params, **self.config.actor_kwargs)
        self.actor_target_network = actor_model(env_params, **self.config.actor_kwargs)
        self.critic_network = critic_model(env_params, **self.config.critic_kwargs)
        self.critic_target_network = critic_model(env_params, **self.config.critic_kwargs)
        if self.config.use_air:
            assert self.config.replay_strategy == 'none'
            self.air = AIR(self.config.gamma, self.env.compute_reward, env_params, \
                self.actor_target_network, self.critic_target_network, self._preproc_inputs)
        # load paramters
        if config.resume:
            if self.config.model_path == None:
                path = os.path.join(self.config.save_dir, self.config.env_name, self.config.name, 'latest_model.pt')
            else:
                path = self.config.model_path
            if self.config.shared_normalizer:
                robot_dict, object_dict, goal_dict, actor_model, critic_model = torch.load(path, map_location=lambda storage, loc: storage)
            else:
                o_dict, g_dict, actor_model, critic_model = torch.load(path, map_location=lambda storage, loc: storage)
            # OLD Version 
            # o_mean, o_std, g_mean, g_std, actor_model, critic_model = torch.load(path, map_location=lambda storage, loc: storage)
            print('loaded done!')
            if self.config.actor_master_slave:
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
        if self.config.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.config.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.config.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.config.replay_strategy, self.config.replay_k, self.env_params['compute_reward'], \
            random_unmoved_rate = self.config.random_unmoved_rate, not_relabel_unmoved = self.config.not_relabel_unmoved)
        # goal sampler
        # self.goal_sampler = goal_sampler()
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.config.buffer_size, self.her_module.sample_her_transitions, store_info = self.config.store_info)
        # create the normalizer
        if self.config.shared_normalizer:
            self.robot_norm = normalizer(size=env_params['robot_obs_size'], default_clip_range=self.config.clip_range)
            self.object_norm = normalizer(size=env_params['obj_obs_size'], default_clip_range=self.config.clip_range)
            self.goal_norm = normalizer(size=env_params['goal_size'], default_clip_range=self.config.clip_range)
            if self.config.resume:
                self.robot_norm.load(robot_dict)
                self.object_norm.load(object_dict)
                self.goal_norm.load(goal_dict)
        else:
            self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.config.clip_range)
            self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.config.clip_range)
            if config.resume:
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
            # if not os.path.exists(self.config.save_dir):
            #     os.mkdir(self.config.save_dir, exist_ok=True)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # start wandb to log
            if self.config.wandb:
                wandb.init(
                    project = self.config.project,
                    group = self.config.group,
                    tags = self.config.tags, 
                    name = self.config.name,
                    notes = f'Env:{self.config.env_name},Note:{self.config.note}', 
                    resume = self.config.resume
                )

    def learn(self):
        """
        train the network

        """
        # warm up
        if self.config.warmup:
            self.warmup(100)
        # start to collect samples
        start_time = time()
        collect_per_epoch = self.config.n_cycles * self.config.num_rollouts_per_mpi * self.env_params['max_timesteps']
        self.global_relabel_rate = 0.3
        curriculum_param = self.config.curriculum_init
        curri_indicator = 0
        best_success_rate = 0
        total_steps = 0
        for epoch in range(self.config.n_epochs):
            # start curriculum
            if self.config.curriculum and curri_indicator > self.config.curriculum_bar:
                if curriculum_param < self.config.curriculum_end:
                    best_success_rate = 0
                    curriculum_param += self.config.curriculum_step
                    path = self.model_path + f'/curr{curriculum_param:.2f}_model.pt'
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        if self.config.shared_normalizer:
                            torch.save([self.robot_norm.state_dict(), self.object_norm.state_dict(), \
                                self.goal_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], path)
                        else:
                            torch.save([self.o_norm.state_dict(), self.g_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], path)
                        if self.config.wandb:
                            wandb.save(path)
                        print(f'save curriculum {curriculum_param:.2f} end model at {self.model_path}')
                observation = self.env.reset({self.config.curriculum_attr: curriculum_param})
                # extend buffer to new observation
                o_size = len(observation['observation'])
                g_size = len(observation['desired_goal'])
                # extend normalizer to new observation
                if not self.config.shared_normalizer:
                    self.o_norm.change_size(new_size = o_size)
                    self.g_norm.change_size(new_size = g_size)
                self.buffer.change_size(max_timesteps=self.env._max_episode_steps,\
                    obs_size=o_size, goal_size=g_size)
            num_useless_rollout = 0 # record number of useless rollout(ag not change)
            for _ in tqdm(range(self.config.n_cycles)):
                mb_obs, mb_ag, mb_g, mb_info, mb_actions = [], [], [], {}, []
                for k in self.config.store_info:
                    mb_info[k] = []
                for _ in range(self.config.num_rollouts_per_mpi):
                    # try until collect successful experience
                    for j in range(self.config.max_trail_time):
                        # reset the rollouts
                        ep_obs, ep_ag, ep_g, ep_info, ep_actions = [], [], [], {}, []
                        for k in self.config.store_info:
                            ep_info[k] = []
                        # reset the environment
                        observation = self.env.reset()
                        obs = observation['observation']
                        ag = observation['achieved_goal']
                        g = observation['desired_goal']
                        info = observation.get('info') # if no info, return None
                        # start to collect samples
                        ag_origin = ag
                        extra_reset_steps = np.random.randint(self.env._max_episode_steps) if self.config.extra_reset_steps and np.random.uniform() < 0.5 else 0
                        delay_agent = np.random.uniform() < 0.5
                        for t in range(self.env._max_episode_steps+extra_reset_steps):
                            if self.config.actor_network == 'actor_attn':
                                input_kwargs = {'mask': info['mask']}
                            else:
                                input_kwargs = {}
                            with torch.no_grad():
                                input_tensor = self._preproc_inputs(obs, g)
                                pi = self.actor_network(input_tensor, **input_kwargs)
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
                                for k in self.config.store_info:
                                    ep_info[k].append(info[k])
                                ep_actions.append(action.copy())
                            # re-assign the observation
                            obs = obs_new
                            ag = ag_new
                        # check if use this rollout
                        if_moved = np.linalg.norm(ag.reshape(-1, self.config.dim) - ag_origin.reshape(-1,self.config.dim), axis=-1) > 0.005
                        if_drop = np.any(ag.reshape(-1,self.config.dim)[..., -1] < (-0.1)) and self.config.ignore_drop
                        if self.config.trail_mode == 'all':
                            if_moved = if_moved.all()
                        elif self.config.trail_mode == 'any':
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
                    for k in self.config.store_info:
                        mb_info[k].append(ep_info[k])
                    if self.config.use_air:
                        ep_g = self.air.get_goal(ep_obs, ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                for k in self.config.store_info:
                    mb_info[k] = np.array(mb_info[k])
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
            data = self._eval_agent(render = ((epoch%10)==0  and self.config.render))
            case_data = {}
            for k,v in self.config.eval_kwargs.items():
                if isinstance(v, list):
                    for j in v:
                        case_data[f'{k}{j:.2f}'] = self._eval_agent(render = ((epoch%10)==0  and self.config.render), eval_kwargs = {k:j})['success_rate']
                else:
                    case_data[k] = self._eval_agent(render = ((epoch%10)==0  and self.config.render), eval_kwargs = {k:v})['success_rate']
            rates = []
            for k,v in case_data.items():
                rates.append(1/(v+0.1))
            if len(case_data) > 0:
                self.other_side_rate = 1 - np.array(rates)[0]/sum(rates)
                data = {**data, **case_data}
            curri_indicator = data[self.config.curriculum_indicator]
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
                if self.config.shared_normalizer:
                    torch.save([self.robot_norm.state_dict(), self.object_norm.state_dict(), \
                        self.goal_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/latest_model.pt')
                else:
                    torch.save([self.o_norm.state_dict(), self.g_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/latest_model.pt')
                if self.config.wandb:
                    wandb.save(self.model_path + '/latest_model.pt')
                if data['success_rate'] > best_success_rate:
                    best_success_rate = data['success_rate']
                    if self.config.shared_normalizer:
                        torch.save([self.robot_norm.state_dict(), self.object_norm.state_dict(), \
                            self.goal_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], \
                                self.model_path + f'/curr{curriculum_param:.2f}_best_model.pt')
                    else:
                        torch.save([self.o_norm.state_dict(), self.g_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], \
                                self.model_path + f'/curr{curriculum_param:.2f}_best_model.pt')
                    if self.config.wandb:
                        wandb.save(self.model_path + f'/curr{curriculum_param:.2f}_best_model.pt')
                    print(f'save curriculum {curriculum_param:.2f} best model at {self.model_path}')
                if self.config.wandb:
                    # log data
                    log_data = {
                        'success rate': data['success_rate'], 
                        "reward": data['reward'], 
                        "curriculum param": curriculum_param, 
                        "run time": (time()-start_time)/3600, 
                        "useless rollout per epoch": num_useless_rollout/(self.config.n_cycles*self.config.num_rollouts_per_mpi),
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
        if self.config.shared_normalizer:
            robot_norm = self.robot_norm.normalize(obs[..., :self.env_params['robot_obs_size']])
            object_norm = np.zeros_like(obs[..., self.env_params['robot_obs_size']:])
            for i in range(self.env.num_blocks):
               object_norm[..., i*self.env_params['obj_obs_size']:(i+1)*self.env_params['obj_obs_size']] = \
                   self.object_norm.normalize(
                   obs[..., self.env_params['robot_obs_size'] + i*self.env_params['obj_obs_size']:\
                       self.env_params['robot_obs_size'] + (i+1)*self.env_params['obj_obs_size']])
            goal_norm = np.zeros_like(g)
            for i in range(self.env.num_blocks):
                goal_norm[..., i*self.env_params['goal_size']:(i+1)*self.env_params['goal_size']] = \
                   self.goal_norm.normalize(obs[...,i*self.env_params['goal_size']:+ (i+1)*self.env_params['goal_size']])
            inputs = np.concatenate((robot_norm, object_norm, goal_norm), axis=-1)
        else:
            obs_norm = self.o_norm.normalize(obs)
            g_norm = self.g_norm.normalize(g)
            # concatenate the stuffs
            inputs = np.concatenate((obs_norm, g_norm), axis=-1)
        if not_unsqueeze:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        else:
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.config.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.config.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.config.random_eps, 1)[0] * (random_actions - action)
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
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       **mb_info, 
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        if self.config.shared_normalizer:
            self.goal_norm.update(transitions['g'].reshape(-1, self.env_params['goal_size']))
            self.robot_norm.update(transitions['obs'][..., :self.env_params['robot_obs_size']])
            self.object_norm.update(transitions['obs'][..., self.env_params['robot_obs_size']:]\
                .reshape(-1, self.env_params['obj_obs_size']))
            self.goal_norm.recompute_stats()
            self.object_norm.recompute_stats()
            self.robot_norm.recompute_stats()
        else:
            # update
            self.o_norm.update(transitions['obs'])
            self.g_norm.update(transitions['g'])
            # recompute the stats
            self.o_norm.recompute_stats()
            self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.config.clip_obs, self.config.clip_obs)
        g = np.clip(g, -self.config.clip_obs, self.config.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.config.polyak) * param.data + self.config.polyak * target_param.data)

    # update the network
    def _update_network(self):
        update_times = int(self.config.update_per_step * self.config.num_rollouts_per_mpi * self.env_params['max_episode_steps'])
        for _ in range(update_times):
            # sample the episodes
            transitions = self.buffer.sample(self.config.batch_size)
            # pre-process the observation and goal
            o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
            mask = transitions.get('mask')
            transitions['obs'], transitions['g'] = self._preproc_og(o, g)
            transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
            # start to do the update
            # obs_norm = self.o_norm.normalize(transitions['obs'])
            # g_norm = self.g_norm.normalize(transitions['g'])
            # inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
            # inputs_norm = self._preproc_inputs(transitions['obs'], transitions['g'], not_unsqueeze=True)
            # obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
            # g_next_norm = self.g_norm.normalize(transitions['g_next'])
            # inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
            # inputs_next_norm = self._preproc_inputs(transitions['obs_next'], transitions['g_next'], not_unsqueeze=True)
            # transfer them into the tensor
            inputs_norm_tensor = self._preproc_inputs(transitions['obs'], transitions['g'], not_unsqueeze=True)
            # inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
            inputs_next_norm_tensor = self._preproc_inputs(transitions['obs_next'], transitions['g_next'], not_unsqueeze=True)
            # inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
            actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
            r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
            if self.config.cuda:
                inputs_norm_tensor = inputs_norm_tensor.cuda()
                inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
                actions_tensor = actions_tensor.cuda()
                r_tensor = r_tensor.cuda()
            if self.config.actor_network == 'actor_attn':
                input_kwargs = {'mask': mask}
            else:
                input_kwargs = {}
            # calculate the target Q value function
            with torch.no_grad():
                # do the normalization
                actions_next = self.actor_target_network(inputs_next_norm_tensor, **input_kwargs)
                q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next, **input_kwargs)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.config.gamma * q_next_value
                target_q_value = target_q_value.detach()
                # clip the q value
                clip_return = 1 / (1 - self.config.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            # the q loss
            real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor, **input_kwargs)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            # the actor loss
            actions_real = self.actor_network(inputs_norm_tensor, **input_kwargs)
            q_loss = -self.critic_network(inputs_norm_tensor, actions_real, **input_kwargs).mean()
            actor_loss = self.config.q_coef * q_loss
            actor_loss += self.config.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
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
        for n in range(self.config.n_test_rollouts):
            per_success_rate = []
            per_reward = []
            observation = self.env.reset(**eval_kwargs)
            obs = observation['observation']
            g = observation['desired_goal']
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    if self.config.actor_network == 'actor_attn':
                        input_kwargs = {'mask': info['mask']}
                    else:
                        input_kwargs = {}
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor, **input_kwargs)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, _, info = self.env.step(actions)
                # self.env.render(mode='human')
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
                per_reward.append(reward)
                if MPI.COMM_WORLD.Get_rank() == 0 and render:
                    frame = np.array(self.env.render(mode = 'rgb_array'))
                    frame = np.moveaxis(frame, -1, 0)
                    video.append(frame)
            total_success_rate.append(per_success_rate)
            total_reward.append(per_reward)
        if MPI.COMM_WORLD.Get_rank() == 0 and render and self.config.wandb:
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
        #     if self.config.wandb and MPI.COMM_WORLD.Get_rank() == 0:
        #         self.new_actor_loss.append(new_actor_loss.detach().cpu())
        #         if len(self.new_actor_loss) == 2000:
        #             wandb.log({"actor loss": np.mean(self.new_actor_loss)})
        #             self.new_actor_loss = []
        # self.actor_network.train()
        exit()