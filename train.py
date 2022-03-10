from logging import exception
import numpy as np
import gym
import os, sys
from utils.arguments import get_config
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
import panda_gym
from vec_env.wrapper import VecPyTorch
from vec_env.subproc_vec_env import SubprocVecEnv
# import gym_xarm, gym_naive, 
# import fetch_block_construction

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(config, env):
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    # get parameter for ReNN (if config not found, use default for fetch env)
    try:
        obj_obs_size = env.task.obj_obs_size
    except:
        obj_obs_size = 15
    try:
        robot_obs_size = env.robot_obs_size
    except:
        robot_obs_size = 10
    try:
        goal_size = env.task.goal_size
    except:
        goal_size = 3
    try:
        num_agents = env.num_agents
    except:
        num_agents = 2
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'obj_obs_size': obj_obs_size, 
            'robot_obs_size': robot_obs_size,
            'ignore_goal_size': 0,
            'goal_size': goal_size,
            'num_agents': num_agents, 
            'dim': config.dim, 
            'max_episode_steps': env._max_episode_steps, 
            'max_timesteps': env._max_episode_steps, 
            'compute_reward': env.compute_reward
            }
    for k in config.store_info:
        params[k] = info[k].shape[0]
    return params

def launch(config):
    # create the ddpg_agent

    if 'formation' in config.env_name:
        import formation_gym
        env = formation_gym.make_env(config.env_name, benchmark=False, num_agents = config.num_agents, reward_type=config.reward_type)
    else:
        if config.num_envs == 1:
            env = gym.make(config.env_name, **config.env_kwargs)
            env.seed(config.seed + MPI.COMM_WORLD.Get_rank())
        else:
            def make_thunk(rank):
                config.env_kwargs['seed'] = (config.seed + rank + MPI.COMM_WORLD.Get_rank())
                return lambda: gym.make(config.env_name, **config.env_kwargs)
            env = SubprocVecEnv([make_thunk(i) for i in range(config.num_workers)])
            env._max_episode_steps = env_params['max_episode_steps']
    env_params = get_env_params(config, env)
    # set random seeds for reproduce
    random.seed(config.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(config.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(config.seed + MPI.COMM_WORLD.Get_rank())
    if config.cuda:
        torch.cuda.manual_seed(config.seed + MPI.COMM_WORLD.Get_rank())
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(config, env, env_params)
    ddpg_trainer.learn()
    env.close()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    config = get_config()
    launch(config)