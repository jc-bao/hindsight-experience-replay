import os
import ctypes
import numpy as np

def gcc_complie(c_path, so_path=None):
	assert c_path[-2:]=='.c'
	if so_path is None:
		so_path = c_path[:-2]+'.so'
	else:
		assert so_path[-3:]=='.so'
	os.system('gcc -o '+so_path+' -shared -fPIC '+c_path+' -O2')
	return so_path

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None, random_unmoved = False, not_relabel_unmoved = False):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.random_unmoved = random_unmoved
        self.not_relabel_unmoved = not_relabel_unmoved
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.replace_rate = []
        self.total_sample_num = 1
        self.relabel_num = 0
        self.random_num = 0
        self.nochange_num = 0

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)[0]
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        if self.not_relabel_unmoved:
            sample_size, goal_dim = future_ag.shape
            num_obj = int(goal_dim/3)
            old_ag = transitions['ag'][her_indexes]
            old_goal = transitions['g'][her_indexes]
            if_done = np.linalg.norm(old_ag.reshape(sample_size, num_obj,3) - old_goal.reshape(sample_size, num_obj,3), axis=-1) < 0.05
            if_moved = np.linalg.norm(future_ag.reshape(sample_size, num_obj,3) - old_ag.reshape(sample_size, num_obj,3), axis=-1) > 0.0005
            relabel_musk = np.logical_and((np.logical_not(if_done)), if_moved).reshape(sample_size, num_obj,-1)
            random_musk = np.logical_and((np.logical_not(if_done)), np.logical_not(if_moved)).reshape(sample_size, num_obj,-1)
            nochange_musk = if_done.reshape(sample_size, num_obj,-1)
            # record parameters
            self.total_sample_num += relabel_musk.size
            self.relabel_num += np.sum(relabel_musk)
            self.random_num += np.sum(random_musk)
            self.nochange_num += np.sum(nochange_musk)
            relabel_musk = np.repeat(relabel_musk, 3, axis=-1).reshape(sample_size, -1)
            random_musk = np.repeat(random_musk, 3, axis=-1).reshape(sample_size, -1)
            nochange_musk = np.repeat(nochange_musk, 3, axis=-1).reshape(sample_size, -1)
            if self.random_unmoved:
                random_goal = np.random.uniform([-0.4, -0.15, 0.02], [0.4, 0.15, 0.2], size=(sample_size, num_obj, 3)).reshape(sample_size, -1)
                new_goal = future_ag*relabel_musk + old_goal*nochange_musk + random_goal*random_musk
            else:
                new_goal = future_ag*relabel_musk + old_goal*np.logical_or(nochange_musk, random_musk) 
        else:
            new_goal = future_ag
        transitions['g'][her_indexes] = new_goal
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims([self.reward_func(transitions['ag_next'][i], transitions['g'][i], transitions['info'][i]) for i in range(len(transitions['g']))], 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
 
        return transitions

# class goal_sampler:
#     def __init__(self, args, env, env_params, hgg_pool_size = 1000):
#         self.args = args
#         self.env = env
#         self.dim = env_params['goal_size']
#         self.delta = self.env.task.distance_threshold # thereshold
# 		self.length = args.n_cycles*args.num_rollouts_per_mpi # goal collected per episode
# 		init_goal = self.env.reset()['achieved_goal'].copy()
#         # use pool to store goal candidate
# 		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
# 		self.init_state = self.env.reset()['observation'].copy()
# 		self.match_lib = gcc_complie('hgg/cost_flow.c')
#         # store achieved goal
# 		self.achieved_goal_pool = GoalPool(hgg_pool_size)

# 		# estimating diameter
# 		self.max_dis = 0
# 		for i in range(1000):
# 			obs = self.env.reset()
# 			dis = goal_distance(obs['achieved_goal'],obs['desired_goal'])
# 			if dis>self.max_dis: self.max_dis = dis

#     def sample(self, idx):
#         return self.add_noise(self.pool[idx])

#     def add_noise(self, pre_goal, noise_std=None):
# 		goal = pre_goal.copy()
# 		dim = 2 if self.args.env[:5]=='Fetch' else self.dim
# 		if noise_std is None: noise_std = self.delta
# 		goal[:dim] += np.random.normal(0, noise_std, size=dim)
# 		return goal.copy()

#     def update(self, initial_goals, desired_goals):
#         '''
#         input: goal_pool, init_state, sampled_goal
#         output: new_goal_pool
#         '''
#         if self.achieved_goal_pol.counter==0:
#             self.pool = desired_goals.copy()
#         else:
#             achieved_pool, achieved_pool_init_state = self.achieved_goal_pol.pad()
#             candidate_goals = [] # candidate goals per trajectory
#             candidate_edges = [] # if use this trajectory
#             candidate_id = []
#             # revaluate Q 
#             agent = self.args.agent
#             achieved_value = []
#             for i in range(len(achieved_pool)):
#                 obs = [ goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for  j in range(achieved_pool[i].shape[0])]
#                 feed_dict = {
#                     agent.raw_obs_ph: obs
#                 }
#                 value = agent.sess.run(agent.q_pi, feed_dict)[:,0]
#                 value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
#                 achieved_value.append(value.copy())

#             n = 0
#             graph_id = {'achieved':[],'desired':[]}
#             for i in range(len(achieved_pool)):
#                 n += 1
#                 graph_id['achieved'].append(n)
#             for i in range(len(desired_goals)):
#                 n += 1
#                 graph_id['desired'].append(n)
#             n += 1
#             self.match_lib.clear(n)

#             for i in range(len(achieved_pool)):
#                 self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
#             for i in range(len(achieved_pool)):
#                 for j in range(len(desired_goals)):
#                     res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1)) - achieved_value[i]/(self.args.hgg_L/self.max_dis/(1-self.args.gamma))
#                     match_dis = np.min(res)+goal_distance(achieved_pool[i][0], initial_goals[j])*self.args.hgg_c
#                     match_idx = np.argmin(res)

#                     edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
#                     candidate_goals.append(achieved_pool[i][match_idx])
#                     candidate_edges.append(edge) # if use this trajectory
#                     candidate_id.append(j)
#             for i in range(len(desired_goals)):
#                 self.match_lib.add(graph_id['desired'][i], n, 1, 0)

#             match_count = self.match_lib.cost_flow(0,n)
#             assert match_count==self.length

#             explore_goals = [0]*self.length
#             for i in range(len(candidate_goals)):
#                 if self.match_lib.check_match(candidate_edges[i])==1:
#                     explore_goals[candidate_id[i]] = candidate_goals[i].copy()
#             assert len(explore_goals)==self.length
#             self.pool = np.array(explore_goals)

# class GoalPool:
#     '''
#     store achieved goal/its initial state
#     '''
# 	def __init__(self, pool_length):
# 		self.length = pool_length
# 		self.pool = []
# 		self.pool_init_state = []
# 		self.counter = 0

# 	def insert(self, trajectory, init_state):
# 		if self.counter<self.length:
# 			self.pool.append(trajectory.copy())
# 			self.pool_init_state.append(init_state.copy())
# 		else:
# 			self.pool[self.counter%self.length] = trajectory.copy()
# 			self.pool_init_state[self.counter%self.length] = init_state.copy()
# 		self.counter += 1

# 	def pad(self):
#         '''
#         paddinng to full length
#         '''
# 		if self.counter>=self.length:
# 			return self.pool.copy(), self.pool_init_state.copy()
# 		pool = self.pool.copy()
# 		pool_init_state = self.pool_init_state.copy()
# 		while len(pool)<self.length:
# 			pool += self.pool.copy()
# 			pool_init_state += self.pool_init_state.copy()
# 		return pool[:self.length].copy(), pool_init_state[:self.length].copy()