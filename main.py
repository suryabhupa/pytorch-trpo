import torch
import argparse
from itertools import count

import gym
from gym import spaces
import scipy.optimize
from copy import copy, deepcopy

from models import *
from replay_memory import FullMemory, RolloutMemory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
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
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--rollout-interval', type=int, default=1000, metavar='N',
                    help='interval between collecting multiple rollouts from each state(default: 3)')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

args.discrete = (type(env.action_space) == spaces.discrete.Discrete)

if args.discrete:
    policy_net = DiscretePolicy(num_inputs, num_actions)
else:
    policy_net = Policy(num_inputs, num_actions)

value_net = Value(num_inputs)

def select_action(state):
		if args.discrete:
				state = torch.from_numpy(state).unsqueeze(0)
				probabilities = policy_net(Variable(state))
				action = probabilities.multinomial(1)
				return action, probabilities
		else:
				state = torch.from_numpy(state).unsqueeze(0)
				action_mean, _, action_std = policy_net(Variable(state))
				action = torch.normal(action_mean, action_std)
				return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    if args.discrete:
        action_probs = torch.Tensor(np.concatenate(batch.action_prob, 0))
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

    def surrogate_loss(self, theta):
        """
        Returns the surrogate loss w.r.t. the given parameter vector theta
        """
        new_model = copy.deepcopy(self.policy_model)
        vector_to_parameters(theta, new_model.parameters())
        observations_tensor = torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in self.observations])
        prob_new = new_model(observations_tensor).gather(1, torch.cat(self.actions)).data
        prob_old = self.policy_model(observations_tensor).gather(1, torch.cat(self.actions)).data + 1e-8
        return -torch.mean((prob_new / prob_old) * self.advantage)

    if args.discrete:
        fixed_probs = policy_net(Variable(states))
    else:
        action_means, action_log_stds, action_stds = policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if args.discrete:
            probs = policy_net(Variable(states, volatile=volatile))
            action_loss = -Variable(advantages) * (probs / fixed_probs)
            return action_loss.mean()
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()


    def get_kl():
        if args.discrete:
            action_probs = policy_net(Variable(states))

        else:
            mean1, log_std1, std1 = policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

for i_episode in count(1):
    memory = FullMemory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0

    print('Beginning collecting samples...')
    if i_episode % args.rollout_interval == 0:
        print('Collecting extra rollouts this episode...')

    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        actions = []
        actions_probs = []
        reward_sum = 0
        for t in range(1000): # Don't infinite loop while learning

            if args.discrete:
                action, action_prob = select_action(state)
                orig_action_prob = action_prob.data[0].numpy()
                actions_probs.append(action_prob)
            else:
                action, orig_action_prob = select_action(state), None

            actions.append(action)
            orig_action = action.data[0].numpy()

            rollout = RolloutMemory()
            # Collect rollouts
            if i_episode % args.rollout_interval == 0 and t % 100 == 0:
                orig_qpos, orig_qvel = env.env.get_state()[0].copy(), env.env.get_state()[1].copy()
                rollout_reward_sum = 0
                rollout_state = state
                rollout_action = orig_action
                rollout_action_prob = orig_action_prob
                for k in range(1000):
                    # Use last rollout state and action
                    if args.discrete:
                        rollout_next_state, rollout_reward, rollout_done, _ = env.step(rollout_action[0])
                    else:
                        rollout_next_state, rollout_reward, rollout_done, _ = env.step(rollout_action)
                    rollout_reward_sum += rollout_reward
                    rollout_next_state = running_state(rollout_next_state)

                    rollout_mask = 1
                    if rollout_done:
                        rollout_mask = 0

                    # Save everything
                    rollout.push(state, np.array([action]), rollout_mask, rollout_next_state, rollout_reward, np.array([rollout_action_prob]))

                    if rollout_done:
                        break

                    # Set rollout state
                    rollout_state = rollout_next_state

                    # Set next action
                    if args.discrete:
                        rollout_action, rollout_action_prob = select_action(rollout_state)
                        rollout_action = rollout_action.data[0].numpy()
                        rollout_action_prob = rollout_action_prob.data[0].numpy()
                    else:
                        rollout_action = select_action(rollout_state)
                        rollout_action = rollout_action.data[0].numpy()
                        rollout_action_prob = None


                env.env.set_state(orig_qpos, orig_qvel)

            if args.discrete:
                next_state, reward, done, _ = env.step(orig_action[0])
            else:
                next_state, reward, done, _ = env.step(orig_action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            # print('orig_action', orig_action)
            # print('np.array([orig_action]', np.array([orig_action]))
            # print('orig_action_prob', orig_action_prob)
            # print('orig_action_prob', np.array([orig_action_prob]))
            memory.push(state, np.array([orig_action]), mask, next_state, reward, rollout, np.array([orig_action_prob]))

            if args.render:
                env.render()
            if done:
                break

            state = next_state

        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    print('Finished collecting samples!')
    reward_batch /= num_episodes
    batch = memory.sample()
    update_params(batch)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
