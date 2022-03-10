import argparse
import json
import yaml
from .misc import dotdict

"""
Here are the param for the training

"""

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default=None, help='config file')
    # rl basic
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    # env
    parser.add_argument('--env-name', type=str, default='PandaRearrangeBimanual-v0', help='the environment name')
    parser.add_argument('--env-kwargs', type=json.loads, default={})
    parser.add_argument('--num-envs', type=int, default=1, help='number of subvec envs')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    # train
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--n-epochs', type=int, default=50, help='big cycle, for save')
    parser.add_argument('--n-cycles', type=int, default=50, help='small cycle, for update')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--update-per-step', type=float, default=0.04, help='the times to update the network')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--q-coef', type=float, default=1.0)
    # save
    parser.add_argument('--resume', action='store_true', help='if resume old model')
    parser.add_argument('--model-path', default=None, type=str)
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    # explore
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--max-trail-time', type=int, default=10, help='max trail time to collect a successful experence')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--trail-mode', type=str, default='any')
    parser.add_argument('--extra-reset-steps', action='store_true')
    # replay
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--use-air', action='store_true')
    parser.add_argument('--random-unmoved-rate', type = float, default = 1)
    parser.add_argument('--not-relabel-unmoved', action='store_true')
    # eval
    parser.add_argument('--eval-kwargs', type=json.loads, default={})
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    # normalizer
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--shared-normalizer', action='store_true')
    # curriculum
    parser.add_argument('--curriculum', action='store_true', help='if use curriculum to train')
    parser.add_argument('--curriculum-indicator', type=str, default='success_rate')
    parser.add_argument('--curriculum-bar', default=0.5, type = float)
    parser.add_argument('--curriculum-init', default=0, type = float)
    parser.add_argument('--curriculum-end', default=1, type = float)
    parser.add_argument('--curriculum-step', default=0.1, type = float)
    # network
    parser.add_argument('--actor-model', type = str, default = 'actor')
    parser.add_argument('--actor-kwargs', type = json.loads, default = {})
    parser.add_argument('--critic-model', type = str, default = 'critic')
    parser.add_argument('--critic-kwargs', type = json.loads, default = {})
    # wandb mode
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type = str, default = 'debug')
    parser.add_argument('--group', type = str, default = 'ungrouped')
    parser.add_argument('--tags', type = str, default = '')
    parser.add_argument('--name', type = str, default = 'noname')
    parser.add_argument('--note', type = str, default = '')
    parser.add_argument('--render', action='store_true')
    # multi-agent
    parser.add_argument('--num-agents', type = int, default = 3)
    parser.add_argument('--dim', type = int, default = 3)
    config = parser.parse_args()

    if not config.config_file is None:
        with open('config/' + config.config_file + '.yaml', "r") as stream:
            try:
                config = dotdict(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
        print('use config file to overwrite command line config.')
    return config