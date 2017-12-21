import argparse
from itertools import count

import gym
import scipy.optimize
import numpy as np

import torch
from models import *
from utils import *
from trpo import *
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
        print('mask_ends', mask_ends - np.roll(mask_ends, 1))
        print('max(mask_ends)', max(mask_ends - np.roll(mask_ends, 1)))
        raise NotImplementedError("The episode lengths don't have the same length.")
    ep_len = int(mask_ends[0])

    # Represents normal returns, and 1-, 2-, 3-, 10-step models.
    new_advs = [0, 0, 0, 0, 0]
    new_advs[0] = advantages
    # print('i', 0, 'new_advs[i+1]', new_advs[0].view(-1, ep_len))
    for i, step in enumerate([1, 2, 3, 10]):
        new_advs[i+1] = advantages
        new_advs[i+1] = new_advs[i+1].view(-1, ep_len)
        new_advs[i+1] = torch.cat([new_advs[i+1][:,step:], torch.zeros(new_advs[i+1].size(0), step)], 1)
        new_advs[i+1] = new_advs[i+1].view(-1, 1)
        new_advs[i+1] = new_advs[i+1] * ((args.gamma * args.tau) ** step)
        # print('i', step, 'new_advs[i+1]', new_advs[i+1].view(-1, ep_len))

    # quit(1)
    # import pdb
    # pdb.set_trace()

    if args.adv_norm:
        advantages = (advantages - advantages.mean()) / advantages.std()

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

def _oracle_shift_by_n(tens, ep_len, n):
    if n == 0:
        return tens
    tens = tens.view(-1, ep_len)
    tens = torch.cat([tens[:,n:], torch.zeros(tens.size(0), n)], 1)
    tens = tens.view(-1, 1)
    return tens

def compute_gradient_estimates_qae(batch, policy_net, value_net, qvalue_net, qevalue_net, args, num_eval_grad_steps, grads_list, writer):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    eps = torch.Tensor(np.concatenate(batch.eps, 0))
    eps = torch.cat([eps[1:,:], torch.zeros(1,eps.size(1))])
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))
    values_q = qvalue_net(Variable(torch.cat([states, actions], 1)))
    values_qe = qevalue_net(Variable(torch.cat([states, actions, eps], 1)))

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
    targets_q = Variable(advantages)

    # Does not work on Walker2d-v1, for example.
    mask_ends = np.argwhere(masks.numpy() == 0).flatten() + 1
    if not all(mask_ends[i] % mask_ends[0] == 0 for i in range(len(mask_ends))):
        raise NotImplementedError
    ep_len = int(mask_ends[0])

    new_advs = [0, 0, 0, 0, 0]
    for i, step in enumerate([0, 1, 2, 3, 10]):
        new_advs[i] = advantages
        new_advs[i] = _oracle_shift_by_n(new_advs[i], ep_len, step)
        new_advs[i] = new_advs[i] * ((args.gamma * args.tau) ** step)

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_oracle_eval_loss(advs, volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advs) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def get_qe_loss(advs, volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)

        new_actions = torch.exp(action_log_stds) * Variable(eps) + action_means
        qvalues = qvalue_net(torch.cat([Variable(states), new_actions], 1))
        qvalues_fixed = qvalues.detach()

        first_term = -Variable(advantages - qvalues_fixed.data) * torch.exp(log_prob - Variable(fixed_log_prob))
        second_term = -qvalues
        action_loss = first_term + second_term

        return action_loss.mean()

    def get_qae_loss(advs, volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)

        new_actions = torch.exp(action_log_stds) * Variable(eps) + action_means
        qevalues = qevalue_net(torch.cat([Variable(states), new_actions, eps], 1))
        qevalues_fixed = qevalues.detach()

        first_term = -Variable(advantages - qevalues_fixed.data) * torch.exp(log_prob - Variable(fixed_log_prob))
        second_term = -qevalues
        action_loss = first_term + second_term

        return action_loss.mean()

    if args.eval_grad:
        returns_arr = new_advs
        returns_arr.extend([advantages, advantages, advantages])
        descs = ['0-step', '1-step', '2-step', '3-step', '10-step', 'qvalue', 'qevalue']
        grads_list = aggregate_or_eval_grads_qae(policy_net, returns_arr, [get_oracle_eval_loss, get_qe_loss, get_qae_loss], num_eval_grad_steps, grads_list, args, writer, descs)

    return grads_list

def compute_gradient_estimates_qe(batch, policy_net, value_net, qvalue_net, qevalue_net, args, num_eval_grad_steps, grads_list, writer):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    eps = torch.Tensor(np.concatenate(batch.eps, 0))
    eps = torch.cat([eps[1:,:], torch.zeros(1,eps.size(1))])
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))
    values_q = qvalue_net(Variable(torch.cat([states, actions], 1)))
    values_qe = qevalue_net(Variable(torch.cat([states, actions, eps], 1)))

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
    targets_q = Variable(advantages)

    # Does not work on Walker2d-v1, for example.
    mask_ends = np.argwhere(masks.numpy() == 0).flatten() + 1
    if not all(mask_ends[i] % mask_ends[0] == 0 for i in range(len(mask_ends))):
        raise NotImplementedError
    ep_len = int(mask_ends[0])

    new_advs = [0, 0, 0, 0, 0]
    for i, step in enumerate([0, 1, 2, 3, 10]):
        new_advs[i] = advantages
        new_advs[i] = _oracle_shift_by_n(new_advs[i], ep_len, step)
        new_advs[i] = new_advs[i] * ((args.gamma * args.tau) ** step)

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_oracle_eval_loss(advs, volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advs) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def get_qe_loss(advs, volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)

        new_actions = torch.exp(action_log_stds) * Variable(eps) + action_means
        qvalues = qvalue_net(torch.cat([Variable(states), new_actions], 1))
        qvalues_fixed = qvalues.detach()

        first_term = -Variable(advantages - qvalues_fixed.data) * torch.exp(log_prob - Variable(fixed_log_prob))
        second_term = -qvalues
        action_loss = first_term + second_term

        return action_loss.mean(), first_term.mean(), second_term.mean()

    if args.eval_grad:
        returns_arr = new_advs
        returns_arr.extend([advantages, advantages, advantages])
        descs = ['0-step', '1-step', '2-step', '3-step', '10-step', 'qvalue', 'qfirst', 'qsecond']
        grads_list = aggregate_or_eval_grads_qe(policy_net, returns_arr, [get_oracle_eval_loss, get_qe_loss], num_eval_grad_steps, grads_list, args, writer, descs)

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
    def get_value_loss(flat_params, net):
        set_flat_params_to(net, torch.Tensor(flat_params))
        for param in net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy()[0], get_flat_grad_from(net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), args=[value_net], maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    if args.adv_norm:
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


def update_params_qe(batch, policy_net, value_net, qvalue_net, qevalue_net, args):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    eps = torch.Tensor(np.concatenate(batch.eps, 0))
    eps = torch.cat([eps[1:,:], torch.zeros(1,eps.size(1))])
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))
    values_q = qvalue_net(Variable(torch.cat([states, actions], 1)))
    values_qe = qevalue_net(Variable(torch.cat([states, actions], 1)))

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
    targets_q = Variable(advantages)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params, net, uid, debug=False):
        set_flat_params_to(net, torch.Tensor(flat_params))
        for param in net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        if uid == 'v':
            values_ = net(Variable(states))
            value_loss = (values_ - targets).pow(2).mean()
        elif uid == 'q':
            values_ = net(Variable(torch.cat([states, actions], 1)))
            value_loss = (values_ - targets_q).pow(2).mean()
        elif uid == 'qe':
            values_ = net(Variable(torch.cat([states, actions, eps], 1)))
            value_loss = (values_ - targets_q).pow(2).mean()

        if debug:
          print('====== QE update params ======')
          print('VALUE_LOSS_{} before L2: {}'.format(uid, value_loss.data[0]))
        # weight decay
        for param in net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        if debug:
          print('VALUE_LOSS_{} after L2: {}'.format(uid, value_loss.data[0]))
        value_loss.backward()
        return (value_loss.data.double().numpy()[0], get_flat_grad_from(net).data.double().numpy())

    get_value_loss(get_flat_params_from(value_net).double().numpy(), value_net, 'v', debug=True)
    flat_params_v, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), args=[value_net, 'v'], maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params_v))
    get_value_loss(get_flat_params_from(qvalue_net).double().numpy(), qvalue_net, 'q', debug=True)
    flat_params_q, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(qvalue_net).double().numpy(), args=[qvalue_net, 'q'], maxiter=25)
    set_flat_params_to(qvalue_net, torch.Tensor(flat_params_q))

    if args.adv_norm:
        advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)

        new_actions = torch.exp(action_log_stds) * Variable(eps) + action_means
        qvalues = qvalue_net(torch.cat([Variable(states), new_actions], 1))
        qvalues_fixed = qvalues.detach()

        action_loss = -Variable(advantages - qvalues_fixed.data) \
                * torch.exp(log_prob - Variable(fixed_log_prob)) \
                - qvalues
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


def update_params_qae(batch, policy_net, value_net, qvalue_net, qevalue_net, args):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    eps = torch.Tensor(np.concatenate(batch.eps, 0))
    eps = torch.cat([eps[1:,:], torch.zeros(1,eps.size(1))])
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))
    values_q = qvalue_net(Variable(torch.cat([states, actions], 1)))
    values_qe = qevalue_net(Variable(torch.cat([states, actions], 1)))

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
    targets_q = Variable(advantages)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params, net, uid, debug=False):
        set_flat_params_to(net, torch.Tensor(flat_params))
        for param in net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        if uid == 'v':
            values_ = net(Variable(states))
            value_loss = (values_ - targets).pow(2).mean()
        elif uid == 'q':
            values_ = net(Variable(torch.cat([states, actions], 1)))
            value_loss = (values_ - targets_q).pow(2).mean()
        elif uid == 'qe':
           #  values_ = net(Variable(torch.cat([states, actions, eps], 1)))
            values_ = net(Variable(torch.cat([states, actions], 1)))
            value_loss = (values_ - targets_q).pow(2).mean()

        if debug:
          print('====== QAE update params ======')
          print('VALUE_LOSS_{} before L2: {}'.format(uid, value_loss.data[0]))
        # weight decay
        for param in net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        if debug:
          print('VALUE_LOSS_{} after L2: {}'.format(uid, value_loss.data[0]))
        value_loss.backward()
        return (value_loss.data.double().numpy()[0], get_flat_grad_from(net).data.double().numpy())

    get_value_loss(get_flat_params_from(value_net).double().numpy(), value_net, 'v', debug=True)
    flat_params_v, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), args=[value_net, 'v'], maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params_v))

    get_value_loss(get_flat_params_from(qvalue_net).double().numpy(), qvalue_net, 'q', debug=True)
    flat_params_q, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(qvalue_net).double().numpy(), args=[qvalue_net, 'q'], maxiter=25)
    set_flat_params_to(qvalue_net, torch.Tensor(flat_params_q))

    get_value_loss(get_flat_params_from(qevalue_net).double().numpy(), qevalue_net, 'qe', debug=True)
    flat_params_qe, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(qevalue_net).double().numpy(), args=[qevalue_net, 'qe'], maxiter=25)
    set_flat_params_to(qevalue_net, torch.Tensor(flat_params_qe))

    if args.adv_norm:
        advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)

        new_actions = torch.exp(action_log_stds) * Variable(eps) + action_means
        qevalues = qevalue_net(torch.cat([Variable(states), new_actions], 1))
        qevalues_fixed = qevalues.detach()

        action_loss = -Variable(advantages - qevalues_fixed.data) \
                * torch.exp(log_prob - Variable(fixed_log_prob)) \
                - qevalues
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

