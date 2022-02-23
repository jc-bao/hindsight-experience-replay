import numpy as np
import torch
'''
Approximate IRL Relabeling (AIR)
'''

class AIR:
    def __init__(self, gamma, reward_func, env_params, actor, critic, _preproc_inputs) -> None:
        self.T = env_params['max_timesteps']
        self.gamma = gamma
        self.gamma_list = np.geomspace(1, self.gamma**(self.T-1), self.T)
        self.reward_func = reward_func
        self.actor = actor
        self.critic = critic
        self._preproc_inputs = _preproc_inputs
        pass

    def get_goal(self, ep_obs, ep_ag):
        ep_ag_next = np.array(ep_ag)[1:, :]
        # rew
        step_rew = self.reward_func(ep_ag_next, ep_ag_next.reshape(self.T, 1, -1).repeat(self.T,1), None) # dim2:goal_idx dim1:step-rew
        # manually get the reward
        delta = np.linalg.norm(ep_ag_next.reshape(self.T, 1, -1).repeat(self.T,1) - ep_ag_next, axis=-1)
        step_rew = -(delta>0.05).astype(float)
        total_rew = np.sum(step_rew * self.gamma_list, axis=-1)
        # state value
        inputs = self._preproc_inputs(np.tile(ep_obs[0], (self.T, 1)), ep_ag_next, not_unsqueeze=True)
        with torch.no_grad():
            state_value = self.critic(inputs, self.actor(inputs)).detach().cpu().numpy().flatten()
        a_hat = total_rew - state_value
        goal_idx = np.argmax(a_hat)
        return np.tile(ep_ag_next[goal_idx], (self.T, 1))