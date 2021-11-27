import numpy as np
# 1 trajectory,5 transitions
episode_batch = {
    'obs': np.arange(0,20).reshape(2,5,2),
    'ag': np.arange(0,10).reshape(2,-1,1),
    'g': np.arange(0,10).reshape(2,-1,1),
    'actions': np.arange(0,10).reshape(2,-1,1),
    'obs_next': np.arange(20,40).reshape(2,5,2),
    'ag_next': np.arange(5,15).reshape(2,-1,1),
}
T = episode_batch['actions'].shape[1] # trajecotory length
rollout_batch_size = episode_batch['actions'].shape[0] # total trajectory number = batchsize*rollout_per_MPI
batch_size = 5 # number of transitions per trajectory
# select which rollouts and which timesteps to be used 
episode_idxs = np.random.randint(0, rollout_batch_size, batch_size) # choose #episode_length trajectory
t_samples = np.random.randint(T, size=batch_size) # choose timestep
transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
# her idx [HER idx means choose which transition]
her_indexes = np.where(np.random.uniform(size=batch_size) < 0.8)[0] # [0, episode length] choose 80% number
future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
future_offset = future_offset.astype(int) # #transi number of future t
future_t = (t_samples + 1 + future_offset)[her_indexes]
# replace goal with achieved goal
future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t-1]
# CHANGE1: only change goal when ag is not same with ag
# transitions['g'][her_indexes] = future_ag
for i in range(len(future_ag)):
    if not (future_ag[i] == transitions['ag'][her_indexes][i]).all():
        transitions['g'][her_indexes[i]] = future_ag[i]
# to get the params to re-compute reward
transitions['r'] = np.expand_dims([(transitions['ag_next'][i] - transitions['g'][i]) for i in range(len(transitions['g']))], 1)
transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}