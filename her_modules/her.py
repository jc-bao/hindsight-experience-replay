import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        # CHANGE1: only change goal when ag is not same with ag
        for i in range(len(future_ag)):
            if not (future_ag[i][:3] == transitions['ag'][her_indexes][i][:3]).all():
                transitions['g'][her_indexes][i][:3] = future_ag[i][:3]
            if not (future_ag[i][3:6] == transitions['ag'][her_indexes][i][3:6]).all():
                transitions['g'][her_indexes][i][3:6] = future_ag[i][3:6]
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims([self.reward_func(transitions['ag_next'][i], transitions['g'][i], None) for i in range(len(transitions['g']))], 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
