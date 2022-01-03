import torch
import os
from datetime import datetime
from time import time
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic, critic_bilinear, critic_sum
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import wandb

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network and target network
        self.actor_network = actor(env_params)
        self.actor_target_network = actor(env_params)
        if args.use_bilinear:
            self.critic_network = critic_bilinear(env_params)
            self.critic_target_network = critic_bilinear(env_params)
        elif args.use_critic_sum:
            self.critic_network = critic_sum(env_params)
            self.critic_target_network = critic_sum(env_params)
        else:
            self.critic_network = critic(env_params)
            self.critic_target_network = critic(env_params)
        # load paramters
        if args.resume:
            if self.args.model_path == None:
                path = os.path.join(self.args.save_dir, self.args.env_name, self.args.name, 'model.pt')
            else:
                path = self.args.model_path
            try:
                o_mean, o_std, g_mean, g_std, actor_model, critic_model = torch.load(path, map_location=lambda storage, loc: storage)
            except:
                print('fail to load the model!')
            print('loaded done!')
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
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        if args.resume:
            self.o_norm.std = o_std
            self.o_norm.mean = o_mean
            self.g_norm.std = g_std
            self.g_norm.mean = g_mean
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            # if not os.path.exists(self.args.save_dir):
            #     os.mkdir(self.args.save_dir, exist_ok=True)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name, self.args.name)
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
        curriculum_param = self.args.curriculum_init
        start_time = time()
        collect_per_epoch = self.args.n_cycles * self.args.num_rollouts_per_mpi * self.env_params['max_timesteps']
        for epoch in range(self.args.n_epochs):
            num_useless_rollout = 0 # record number of useless rollout(ag not change)
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # try until collect successful experience
                    for j in range(self.args.max_trail_time):
                        # reset the rollouts
                        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                        # reset the environment
                        observation = self.env.reset()
                        obs = observation['observation']
                        ag = observation['achieved_goal']
                        g = observation['desired_goal']
                        # start to collect samples
                        ag_origin = ag
                        for t in range(self.env_params['max_timesteps']):
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
                        # check if use this rollout
                        if np.sum(abs(ag - ag_origin))>0.01:
                            break
                        else:
                            num_useless_rollout += 1
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
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                if self.args.dynamic_batch: # update according to buffer size
                    update_times = int(self.args.n_batches * self.buffer.current_size / self.buffer.size)
                else:
                    update_times = self.args.n_batches
                for _ in range(update_times):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            data = self._eval_agent(render = ((epoch%10)==0  and self.args.render))
            if self.args.curriculum_reward:
                curri_param = data['reward']
            else:
                curri_param = data['success_rate']
            if self.args.curriculum and curri_param > self.args.curriculum_bar:
                if curriculum_param < self.args.curriculum_end: 
                    curriculum_param += 0.1
                self.env.change(curriculum_param)
                if self.args.use_critic_sum:
                    if self.critic_network.num_goal < self.critic_network.num_obj:
                        self.critic_network.num_goal += 1
                        self.critic_target_network.num_goal += 1
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"same_side_rate: {curriculum_param-0.1} -> {curriculum_param}")
            if MPI.COMM_WORLD.Get_rank() == 0 and self.args.wandb:
                # save data
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, reward is: {:.3f}'.format(datetime.now(), epoch, data['success_rate'], data['reward']))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/model.pt')
                # log data
                wandb.log(
                    {
                        'success rate': data['success_rate'], 
                        "reward": data['reward'], 
                        "curriculum param": curriculum_param, 
                        "run time": (time()-start_time)/3600, 
                        "useless rollout per epoch": num_useless_rollout/(self.args.n_cycles*self.args.num_rollouts_per_mpi),
                        "future relabel rate": self.her_module.relabel_num/self.her_module.total_sample_num, 
                        "random relabel rate": self.her_module.random_num/self.her_module.total_sample_num, 
                        "not change relabel rate": self.her_module.nochange_num/self.her_module.total_sample_num, 
                    }, 
                    step=epoch*collect_per_epoch
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
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
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
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
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
        for n in range(self.args.n_test_rollouts):
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
        if MPI.COMM_WORLD.Get_rank() == 0 and render and self.args.wandb:
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
            # reset the environment
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
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])