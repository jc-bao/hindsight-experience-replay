import torch
import os
from datetime import datetime
from time import time
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, actor_bilinear, critic, critic_bilinear, critic_sum
from rl_modules.renn_models import actor_ReNN, critic_ReNN
from rl_modules.attn_models import actor_attn, critic_attn
from rl_modules.ma_models import actor_shared, actor_separated
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
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
        # MPI
        self.comm = MPI.COMM_WORLD
        self.nprocs = self.comm.Get_size()
        # create the network and target network
        if config.actor_shared:
            self.actor_network = actor_shared(env_params)
            self.actor_target_network = actor_shared(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        elif config.actor_separated:
            self.actor_network = actor_separated(env_params)
            self.actor_target_network = actor_separated(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        elif config.use_renn:
            self.actor_network = actor_ReNN(env_params)
            self.actor_target_network = actor_ReNN(env_params)
            self.critic_network = critic_ReNN(env_params)
            self.critic_target_network = critic_ReNN(env_params)
        elif config.use_bilinear:
            self.actor_network = actor_bilinear(env_params)
            self.actor_target_network = actor_bilinear(env_params)
            self.critic_network = critic_bilinear(env_params)
            self.critic_target_network = critic_bilinear(env_params)
        elif config.use_critic_sum:
            self.actor_network = actor(env_params)
            self.actor_target_network = actor(env_params)
            self.critic_network = critic_sum(env_params)
            self.critic_target_network = critic_sum(env_params)
        elif config.use_attn:
            self.actor_network = actor_attn(env_params)
            self.actor_target_network = actor_attn(env_params)
            self.critic_network = critic_attn(env_params)
            self.critic_target_network = critic_attn(env_params)
        else:
            self.actor_network = actor(env_params)
            self.actor_target_network = actor(env_params)
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        # load paramters
        if config.resume:
            if self.config.model_path == None:
                path = os.path.join(self.config.save_dir, self.config.env_name, self.config.name, 'model.pt')
            else:
                path = self.config.model_path
            try:
                o_dict, g_dict, actor_model, critic_model = torch.load(path, map_location=lambda storage, loc: storage)
            except:
                print('fail to load the model!')
            print('loaded done!')
            if not self.config.not_resume_actor:
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
        self.her_module = her_sampler(self.config.replay_strategy, self.config.replay_k, self.env.compute_reward, random_unmoved = self.config.random_unmoved, not_relabel_unmoved = self.config.not_relabel_unmoved)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.config.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.config.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.config.clip_range)
        if config.resume:
            # Note: if use object number curriculum, the normalizer need to be extended
            self.o_norm.load(o_dict)
            self.g_norm.load(g_dict)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            # if not os.path.exists(self.config.save_dir):
            #     os.mkdir(self.config.save_dir, exist_ok=True)
            # path to save the model
            self.model_path = os.path.join(self.config.save_dir, self.config.env_name, self.config.name)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # start wandb to log
            if self.config.wandb:
                wandb.init(
                    project = self.config.project,
                    group = self.config.group,
                    tags = self.config.tags, 
                    name = self.config.name,
                    notes = f'Env:{self.config.env_name},Note:{self.config.note}'
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
        global_relabel_rate = 0.3
        curriculum_param = self.config.curriculum_init
        curri_indicator = 0
        for epoch in range(self.config.n_epochs):
            # start curriculum
            if self.config.curriculum and curri_indicator > self.config.curriculum_bar:
                if curriculum_param < self.config.curriculum_end:
                    curriculum_param += self.config.curriculum_step
                self.env.change(curriculum_param)
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
            for _ in tqdm(range(self.config.n_cycles)):
                mb_obs, mb_ag, mb_g, mb_info, mb_actions = [], [], [], [], []
                for _ in range(self.config.num_rollouts_per_mpi):
                    # try until collect successful experience
                    for j in range(self.config.max_trail_time):
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
                        for t in range(self.env._max_episode_steps):
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
                            ep_info.append(info.copy())
                            ep_actions.append(action.copy())
                            # re-assign the observation
                            obs = obs_new
                            ag = ag_new
                        # check if use this rollout
                        if_moved = np.linalg.norm(ag.reshape(-1,self.config.dim) - ag_origin.reshape(-1,self.config.dim), axis=-1) > 0.005
                        if self.config.trail_mode == 'all':
                            if_moved = if_moved.all()
                        elif self.config.trail_mode == 'any':
                            if_moved = if_moved.any()
                        else:
                            raise NotImplementedError
                        if if_moved:
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
                if self.config.dynamic_batch: # update according to buffer size
                    update_times = int(self.config.n_batches * self.buffer.current_size / self.buffer.size)
                elif self.config.her_batch:
                    update_times = int(self.config.n_batches / global_relabel_rate)
                else:
                    update_times = self.config.n_batches
                for _ in range(update_times):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            data = self._eval_agent(render = ((epoch%10)==0  and self.config.render))
            if self.config.curriculum_reward:
                curri_indicator = data['reward']
            else:
                curri_indicator = data['success_rate']
            # record relabel rate
            local_relabel_rate = self.her_module.relabel_num/self.her_module.total_sample_num
            local_random_relabel_rate = self.her_module.random_num/self.her_module.total_sample_num
            local_not_relabel_rate = self.her_module.nochange_num/self.her_module.total_sample_num
            local_data = np.array([local_relabel_rate, local_random_relabel_rate, local_not_relabel_rate])
            global_data = np.zeros(3)
            self.comm.Allreduce(local_data, global_data, op=MPI.SUM)
            global_relabel_rate, global_random_relabel_rate, global_not_relabel_rate = global_data/self.nprocs
            # local
            if MPI.COMM_WORLD.Get_rank() == 0:
                # save data
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, reward is: {:.3f}'.format(datetime.now(), epoch, data['success_rate'], data['reward']))
                torch.save([self.o_norm.state_dict(), self.g_norm.state_dict(), self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/model.pt')
                if self.config.wandb:
                    # log data
                    wandb.log(
                        {
                            'success rate': data['success_rate'], 
                            "reward": data['reward'], 
                            "curriculum param": curriculum_param, 
                            "run time": (time()-start_time)/3600, 
                            "useless rollout per epoch": num_useless_rollout/(self.config.n_cycles*self.config.num_rollouts_per_mpi),
                            "future relabel rate": global_relabel_rate, 
                            "random relabel rate": global_random_relabel_rate, 
                            "not change relabel rate": global_not_relabel_rate, 
                        }, 
                        step=(epoch+1)*collect_per_epoch
                    )
            # reset record parameters
            self.her_module.total_sample_num = 1
            self.her_module.relabel_num = 0
            self.her_module.random_num = 0
            self.her_module.nochange_num = 0

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.config.cuda:
            inputs = inputs.cuda()
        return inputs

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
        o = np.clip(o, -self.config.clip_obs, self.config.clip_obs)
        g = np.clip(g, -self.config.clip_obs, self.config.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.config.polyak) * param.data + self.config.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.config.batch_size)
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
        if self.config.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.config.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.config.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
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
    def _eval_agent(self, render = False):
        total_success_rate = []
        total_reward = []
        # record video
        if MPI.COMM_WORLD.Get_rank() == 0:
            video = []
        for n in range(self.config.n_test_rollouts):
            per_success_rate = []
            per_reward = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, _, info = self.env.step(actions)
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
            wandb.log({"video": wandb.Video(np.array(video), fps=30, format="mp4")})
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
    
    def warmup(self, num_rollout):
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(num_rollout):
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environmentz
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            for t in range(self.env_params['max_timesteps']):
                action = self.env.action_space.sample()
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
        self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self._update_normalizer([mb_obs, mb_ag, mb_g , mb_actions])