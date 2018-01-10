import torch
from torch import optim

import argparse
import csv
import os
import datetime
import math
from itertools import count
import pdb

import gym
import scipy.optimize

from models import *
from loss import *
from utils import *

from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--test', action='store_true',
                    help='debug flag; log to test files')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--eval-grad', action='store_true',
                    help='evaluate gradient variance')
parser.add_argument('--eval-grad-gae', action='store_true',
                    help='evaluate gradient variance with GAE')
parser.add_argument('--eval-grad-qe', action='store_true',
                    help='evaluate gradient variance with Q models')
parser.add_argument('--eval-grad-fqe', action='store_true',
                    help='evaluate gradient variance with factorized action baseline')
parser.add_argument('--eval-grad-qae', action='store_true',
                    help='evaluate gradient variance with QE models')
parser.add_argument('--eval-grad-freq', type=int, default=5, metavar='N',
                    help='frequency of gradient variance estimation')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--lr', type=float, default=5e-2, metavar='G',
                    help='learning rate for qvalue_net (default: 1e-3)')
parser.add_argument('--hid-dim', type=int, default=64, metavar='N',
                    help='hidden dimension of Q value network')
parser.add_argument('--value2', action='store_true',
                    help='use alternative network architecture for Qvalue net')
parser.add_argument('--anneal-gamma', type=float, default=0, metavar='G',
                    help='anneal gamma from value to 0.99 over 250 steps (default: 0)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--q-l2-reg', type=float, default=1e-1, metavar='G',
                    help='l2 regularization regression for fitting Q model (default: 1e-1)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--adv-norm', action='store_true',
                    help='normalize advantages')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='batch size (default: 15000)')
parser.add_argument('--max-steps', type=int, default=9000000, metavar='N',
                    help='max steps (default: 9000000)')
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
if args.value2:
  qvalue_net = Value2(num_inputs, num_actions, hid_dim=args.hid_dim)
else:
  qvalue_net = Value(num_inputs=num_inputs + num_actions, hid_dim=args.hid_dim)
q_optim = optim.Adam(qvalue_net.parameters(), lr=args.lr)
qevalue_net = Value(num_inputs + num_actions)# + num_actions)


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    eps = Variable(torch.normal(torch.zeros(action_mean.size()), torch.ones(action_std.size())))
    action = action_mean + eps * action_std
    return eps, action


running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

num_grad_eval_steps = 0

if args.value2 and not args.eval_grad_fqe:
  raise NotImplementedError("Value2 isn't ready for use for other estimators!")

# Preparing files for logging
timestamp = '{:%Y%m%d-%H%M}'.format(datetime.datetime.now())
if args.adv_norm:
    advnorm = "yan"
else:
    advnorm = "nan"

if args.seed == 543:
    seedstr = ""
else:
    seedstr = "_seed-{}".format(args.seed)

if not args.eval_grad_qe:
    l2str = ""
else:
    l2str = "_q-l2-reg-{}".format(args.q_l2_reg)

# Add learning rate to filename
l2str += "_lr-{}".format(args.lr)

# Add hidden dim to filename
l2str += "_hid-dim-{}".format(args.hid_dim)

if not args.batch_size == 15000:
  l2str += "_batch-size-{}".format(args.batch_size)

if args.value2:
  l2str += "_value2"

if args.anneal_gamma != 0:
  l2str += "_anneal-gamma-{}".format(args.anneal_gamma)

if args.eval_grad_gae:
    grads_list = [[], [], [], [], []]
    if args.eval_grad:
        filename = "logs/qe_oracle_gae_{}_eg-freq-{}_{}_{}{}{}.csv".format(args.env_name, args.eval_grad_freq, advnorm, timestamp, seedstr, l2str)
    else:
        filename = "logs/qe_oracle_gae_{}_eg-{}_{}{}{}.csv".format(args.env_name, advnorm, timestamp, seedstr, l2str)
elif args.eval_grad_fqe:
    grads_list = [[], [], [], [], [], [], [], []]
    if args.eval_grad:
        filename = "logs/qe_oracle_fqe_{}_eg-freq-{}_{}_{}{}{}.csv".format(args.env_name, args.eval_grad_freq, advnorm, timestamp, seedstr, l2str)
    else:
        filename = "logs/qe_oracle_fqe_{}_eg-{}_{}{}{}.csv".format(args.env_name, advnorm, timestamp, seedstr, l2str)
elif args.eval_grad_qe:
    grads_list = [[], [], [], [], [], [], [], []]
    if args.eval_grad:
        filename = "logs/qe_oracle_qe_{}_eg-freq-{}_{}_{}{}{}.csv".format(args.env_name, args.eval_grad_freq, advnorm, timestamp, seedstr, l2str)
    else:
        filename = "logs/qe_oracle_qe_{}_eg-{}_{}{}{}.csv".format(args.env_name, advnorm, timestamp, seedstr, l2str)
elif args.eval_grad_qae:
    grads_list = [[], [], [], [], [], [], []]
    if args.eval_grad:
        filename = "logs/qe_oracle_qae_{}_eg-freq-{}_{}_{}{}.csv".format(args.env_name, args.eval_grad_freq, advnorm, timestamp, seedstr)
    else:
        filename = "logs/qe_oracle_qae_{}_eg-{}_{}{}.csv".format(args.env_name, advnorm, timestamp, seedstr)
else:
    grads_list = [[], [], [], [], [], []]
    if args.eval_grad:
        filename = "logs/qe_oracle_{}_eg-freq-{}_{}_{}{}.csv".format(args.env_name, args.eval_grad_freq, advnorm, timestamp, seedstr)
    else:
        filename = "logs/qe_oracle_{}_eg-{}_{}{}.csv".format(args.env_name, advnorm, timestamp, seedstr)

if args.test:
    filename = "logs/test.csv"

file_h = open(filename, "a+")
writer = file_h

if args.eval_grad:
    if args.eval_grad_gae:
        f_cge = compute_gradient_estimates_advs
        writer.write('episode,last reward,average reward,step0,step1,step2,step3,step10\n')
    elif args.eval_grad_qae:
        f_cge = compute_gradient_estimates_qae
        writer.write('episode,last reward,average reward,step0,step1,step2,step3,step10,qmodel,qemodel\n')
    elif args.eval_grad_qe:
        f_cge = compute_gradient_estimates_qe
        writer.write('episode,last reward,average reward,step0,step1,step2,step3,step10,qmodel\n')
    elif args.eval_grad_fqe:
        raise NotImplementedError("compute_gradient_estimates_fqe not yet implemented!")
        f_cge = compute_gradient_estimates_fqe
        writer.write('episode,last reward,average reward,step0,step1,step2,step3,step10,qmodel\n')
    else:
        f_cge = compute_gradient_estimates
        writer.write('episode,last reward,average reward,step0,stephalf,step1,step2,step3,step10\n')

    print("f_cge", f_cge)

if args.anneal_gamma != 0:
    args.gamma = args.anneal_gamma
    gamma_anneal_rate = math.pow(0.99 / args.anneal_gamma, 1./100)


i_episode = 1
while i_episode < int(args.max_steps / args.batch_size):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            eps, action = select_action(state)
            action = action.data[0].numpy()
            eps = eps.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0
            memory.push(state, np.array([eps]), np.array([action]), mask, next_state, reward)

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

    # Switch between gradient evaluation and normal
    if args.eval_grad:
        num_grad_eval_steps += 1
        numer = num_grad_eval_steps % args.eval_grad_freq if num_grad_eval_steps % args.eval_grad_freq else args.eval_grad_freq
        print('Estimating gradient update ({}/{} done)...'.format(numer, args.eval_grad_freq))
        if args.eval_grad_qe or args.eval_grad_qae or args.eval_grad_fqe:
            grads_list = f_cge(batch, policy_net, value_net, qvalue_net, qevalue_net, args, num_grad_eval_steps, grads_list, writer)
        else:
            grads_list = f_cge(batch, policy_net, value_net, args, num_grad_eval_steps, grads_list, writer)

    if num_grad_eval_steps % args.eval_grad_freq == 0 or args.eval_grad == False:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
        if args.eval_grad == False:
            writer.write("{},{},{}\n".format(i_episode, reward_sum, reward_batch))
        else:
            writer.write("{},{},{},".format(i_episode, reward_sum, reward_batch))
        print('Updating policy and value networks...')
        if args.eval_grad_qe:
            update_params_qe(batch, policy_net, value_net, qvalue_net, qevalue_net, args)
        elif args.eval_grad_qae:
            update_params_qae(batch, policy_net, value_net, qvalue_net, qevalue_net, args)
        elif args.eval_grad_fqe:
            update_params_fqe(batch, policy_net, value_net, qvalue_net, qevalue_net, q_optim, args)
        else:
            update_params(batch, policy_net, value_net, args)
        writer.flush()
        os.fsync(writer)
        i_episode += 1

    if i_episode % 5 == 0 and args.anneal_gamma != 0:
        print("ANNEALING GAMMA! Previous: {}, New: {}".format(args.gamma, args.gamma * gamma_anneal_rate))
        args.gamma = args.gamma * gamma_anneal_rate

writer.close()
