import numpy as np
from numpy.core.fromnumeric import mean
import wandb

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
            if_moved = np.linalg.norm(future_ag.reshape(sample_size, num_obj,3) - old_ag.reshape(sample_size, num_obj,3), axis=-1) > 0.05
            relabel_musk = np.logical_and((np.logical_not(if_done)), if_moved).reshape(sample_size, num_obj,-1)
            random_musk = np.logical_and((np.logical_not(if_done)), np.logical_not(if_moved)).reshape(sample_size, num_obj,-1)
            nochange_musk = if_done.reshape(sample_size, num_obj,-1)
            relabel_musk = np.repeat(relabel_musk, 3, axis=-1).reshape(sample_size, -1)
            random_musk = np.repeat(random_musk, 3, axis=-1).reshape(sample_size, -1)
            nochange_musk = np.repeat(nochange_musk, 3, axis=-1).reshape(sample_size, -1)
            if True:
                random_goal = np.random.uniform([-0.4, -0.15, 0.02], [0.4, 0.15, 0.2], size=(sample_size, num_obj, 3)).reshape(sample_size, -1)
                new_goal = future_ag*relabel_musk + old_goal*nochange_musk + random_goal*random_musk
            else:
                new_goal = future_ag*relabel_musk + old_goal*np.logical_or(nochange_musk, random_musk) 
        # CHANGE1: only change goal when ag is not same with ag
        # if self.not_relabel_unmoved:
        #     for i in range(len(future_ag)):
        #         if not (future_ag[i][:3] == transitions['ag'][her_indexes[i]][:3]).all():
        #             transitions['g'][her_indexes[i]][:3] = future_ag[i][:3]
        #         elif self.random_unmoved:
        #             if transitions['g'][her_indexes[i]][0] > 0: 
        #                 transitions['g'][her_indexes[i]][:3] = np.random.uniform([0.1, -0.18], [0.3, 0.18])
        #             else: 
        #                 transitions['g'][her_indexes[i]][:3] = np.random.uniform([-0.3, -0.18], [-0.1, 0.18])
        #         if not (future_ag[i][3:6] == transitions['ag'][her_indexes[i]][3:6]).all():
        #             transitions['g'][her_indexes[i]][3:6] = future_ag[i][3:6]
        #         else:
        #             if transitions['g'][her_indexes[i]][0] > 0: 
        #                 transitions['g'][her_indexes[i]][3:6] = np.random.uniform([0.1, -0.18], [0.3, 0.18])
        #             else: 
        #                 transitions['g'][her_indexes[i]][3:6] = np.random.uniform([-0.3, -0.18], [-0.1, 0.18])
        # else:
        #     # replace goal with achieved goal
        transitions['g'][her_indexes] = new_goal
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims([self.reward_func(transitions['ag_next'][i], transitions['g'][i], None) for i in range(len(transitions['g']))], 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
 
        return transitions