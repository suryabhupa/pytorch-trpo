import argparse
from itertools import count

import gym
import scipy.optimize
import numpy as np

import torch
from models import *
from utils import *
from trpo import trpo_step, aggregate_or_eval_grads
from torch.autograd import Variable


def compute_gradient_estimates_advs(batch, policy_net, value_net, args, num_eval_grad_steps, grads_list, writer):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    mask_ends = np.argwhere(masks.numpy() == 0).flatten() + 1
    if not all(mask_ends[i] % mask_ends[0] == 0 for i in range(len(mask_ends))):
        raise NotImplementedError
    ep_len = int(mask_ends[0])

    # Represents normal returns, R - V, and 1-, 2-, 3-, 10-step models.
    new_advs = [0, 0, 0, 0, 0]
    new_advs[0] = advantages
    for i, step in enumerate([1, 2, 3, 10]):
        new_advs[i+1] = advantages
        new_advs[i+1] = new_advs[i+1].view(-1, ep_len)
        new_advs[i+1] = torch.cat([new_advs[i+1][:,step:], torch.zeros(new_advs[i+1].size(0), step)], 1)
        new_advs[i+1] = new_advs[i+1].view(-1, 1)
        new_advs[i+1] = new_advs[i+1] * ((args.gamma * args.tau) ** step)

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_eval_loss(advs, volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advs) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    # Compute loss with returns / V / Q(s, a) / Q(s, a, e)

    if args.eval_grad:
        grads_list = aggregate_or_eval_grads(policy_net, new_advs, get_eval_loss, num_eval_grad_steps, grads_list, args, writer)

    return grads_list


def compute_gradient_estimates(batch, policy_net, value_net, args, num_eval_grad_steps, grads_list, writer):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    mask_ends = np.argwhere(masks.numpy() == 0).flatten() + 1
    if not all(mask_ends[i] % mask_ends[0] == 0 for i in range(len(mask_ends))):
        raise NotImplementedError
    ep_len = int(mask_ends[0])

    # Represents normal returns, R - V, and 1-, 2-, 3-, 10-step models.
    new_returns = [0, 0, 0, 0, 0, 0]
    new_returns[0] = returns
    new_returns[1] = returns - values.data
    for i, step in enumerate([1, 2, 3, 10]):
        new_returns[i+2] = returns - values.data
        new_returns[i+2] = new_returns[i+2].view(-1, ep_len)
        new_returns[i+2] = torch.cat([new_returns[i+2][:,step:], torch.zeros(new_returns[i+2].size(0), step)], 1)
        new_returns[i+2] = new_returns[i+2].view(-1, 1)
        new_returns[i+2] = new_returns[i+2] * (args.gamma ** step)

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_eval_loss(returns, volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(returns) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    # Compute loss with returns / V / Q(s, a) / Q(s, a, e)

    if args.eval_grad:
        grads_list = aggregate_or_eval_grads(policy_net, new_returns, get_eval_loss, num_eval_grad_steps, grads_list, args, writer)

    return grads_list


def update_params(batch, policy_net, value_net, args):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)
    # Reuse these targets when training each of the value functions

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy()[0], get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    # Compute loss with returns / V / Q(s, a) / Q(s, a, e)

    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

