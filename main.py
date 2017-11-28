import argparse
import csv
import os
import datetime
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from loss import compute_gradient_estimates, update_params

from utils import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--eval-grad', action='store_true',
                    help='evaluate gradient variance')
parser.add_argument('--eval-grad-freq', type=int, default=5, metavar='N',
                    help='frequency of gradient variance estimation')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action


running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

num_grad_eval_steps = 0
grads_list = [[], [], [], []]

# Preparing files for logging
timestamp = '{:%Y%m%d-%H%M}'.format(datetime.datetime.now())
filename = "logs/qe_oracle_{}_eg-freq-{}_{}.csv".format(args.env_name, args.eval_grad_freq, timestamp)
file_h = open(filename, "a+")
# writer = csv.writer(file_h, delimiter=',')
writer = file_h
writer.write('episode,last reward,average reward,step0,step1,step2,step3\n')

for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    batch = memory.sample()

    num_grad_eval_steps += 1

    # Switch between gradient evaluation and normal
    numer = num_grad_eval_steps % args.eval_grad_freq if num_grad_eval_steps % args.eval_grad_freq else args.eval_grad_freq
    print('Estimating gradient update ({}/{} done)...'.format(numer, args.eval_grad_freq))
    grads_list = compute_gradient_estimates(batch, policy_net, value_net, args, num_grad_eval_steps, grads_list, writer)

    if num_grad_eval_steps % args.eval_grad_freq == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
        writer.write("{},{},{},".format(i_episode, reward_sum, reward_batch))
        print('Updating policy and value networks...')
        update_params(batch, policy_net, value_net, args)
        writer.flush()
        os.fsync(writer)

writer.close()
