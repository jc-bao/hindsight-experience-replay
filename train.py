from logging import exception
import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
import gym_xarm, gym_naive, panda_gym
import fetch_block_construction

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
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
            'max_timesteps': env._max_episode_steps,
            'obj_obs_size': obj_obs_size, 
            'robot_obs_size': robot_obs_size,
            'ignore_goal_size': 0,
            'goal_size': goal_size,
            'num_agents': num_agents, 
            'dim': args.dim, 
            'drop_out_rate': args.drop_out_rate
            }
    return params

def launch(args):
    # create the ddpg_agent
    # config = {
    #     'GUI': False,
    #     'num_obj': 2, 
    #     'same_side_rate': 0.5, 
    #     'goal_shape': 'any', 
    #     'use_stand': False, 
    # }
    # env = gym.make('XarmHandover-v0', config = config)
    if 'formation' in args.env_name:
        import formation_gym
        env = formation_gym.make_env(args.env_name, benchmark=False, num_agents = args.num_agents, reward_type=args.reward_type)
    else:
        env = gym.make(args.env_name, **args.env_kwargs)
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()
    env.close()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)