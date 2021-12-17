import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
import gym_naive, gym_xarm, panda_gym
import time

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    # model_path = args.save_dir + args.env_name + '/model.pt'
    model_path = '/Users/reedpan/Downloads/model.pt'
    o_mean, o_std, g_mean, g_std, model, _ = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    # config = {
    #     'goal_shape': 'ground', 
    #     'num_obj': 2,
    #     'GUI': False, 
    #     'same_side_rate': 0.0,
    #     'use_stand': False,
    #     'lego_length': 0.15
    # }
    env = gym.make(args.env_name, render = True
    )
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    observation = env.reset()
    # start to do the demo
    obs = observation['observation']
    g = observation['desired_goal']
    # for t in range(env._max_episode_steps):
    while True:
        inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
        with torch.no_grad():
            pi = actor_network(inputs)
        action = pi.detach().numpy().squeeze()
        # put actions into the environment
        observation_new, reward, done , info = env.step(action)
        if done: 
            print('is success: {}'.format(info['is_success']))
            observation_new = env.reset()
            g = observation_new['desired_goal']
        obs = observation_new['observation']
        time.sleep(0.03)