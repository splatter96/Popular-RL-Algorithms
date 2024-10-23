"""
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf

Discrete version reference:
https://towardsdatascience.com/adapting-soft-actor-critic-for-discrete-action-spaces-a20614d4a50a
"""

import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt
import argparse
from common.buffers import *
from foldedtensor import as_folded_tensor

GPU = True
device_idx = 0
if GPU:
    device = torch.device(
        "cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu"
    )
else:
    device = torch.device("cpu")
print(device)


parser = argparse.ArgumentParser(
    description="Train or test neural net motor controller."
)
parser.add_argument("--train", dest="train", action="store_true", default=False)
parser.add_argument("--test", dest="test", action="store_true", default=False)

args = parser.parse_args()


def ints_to_tensor(ints):
    """
    Converts a nested list of integers to a padded tensor.
    """
    if isinstance(ints, torch.Tensor):
        return ints
    if isinstance(ints, list):
        if isinstance(ints[0], (int, bool)):
            return torch.LongTensor(ints)
        if isinstance(ints[0], torch.Tensor):
            return pad_tensors(ints)
        if isinstance(ints[0], list):
            return ints_to_tensor([ints_to_tensor(inti) for inti in ints])


def floats_to_tensor(floats):
    """
    Converts a nested list of floats to a padded tensor.
    """
    if isinstance(floats, torch.Tensor):
        return floats
    if isinstance(floats, list):
        if isinstance(floats[0], (float, np.ndarray, bool)):
            return torch.FloatTensor(floats)
        if isinstance(floats[0], torch.Tensor):
            return pad_tensors(floats)
        if isinstance(floats[0], list):
            return floats_to_tensor([floats_to_tensor(floati) for floati in floats])


def bools_to_tensor(bools):
    """
    Converts a nested list of bools to a padded tensor.
    """
    if isinstance(bools, torch.Tensor):
        return bools
    if isinstance(bools, list):
        print(type(bools[0]))
        if isinstance(bools[0], (bool, np.ndarray)):
            return torch.BoolTensor(bools)
        if isinstance(bools[0], torch.Tensor):
            return pad_tensors(bools)
        if isinstance(bools[0], list):
            return bools_to_tensor([bools_to_tensor(booli) for booli in bools])


def pad_tensors(tensors):
    """
    Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

    The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
    where `Si` is the maximum value of dimension `i` amongst all tensors.
    """
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.zeros(padded_dim)
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, : size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, : size[0], : size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, : size[0], : size[1], : size[2]] = tensor
        else:
            raise ValueError("Padding is supported for upto 3D tensors at max.")
    return padded_tensor


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            np.stack, zip(*batch)
        )  # stack for each element
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, num_actions)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        x = self.linear4(x)
        return x


class SoftQNetworkLSTM(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetworkLSTM, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, num_actions)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, hidden_in):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1, 0, 2)

        x = F.tanh(self.linear1(state))
        x, lstm_hidden = self.lstm1(x, hidden_in)  # no activation after lstm
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        x = self.linear4(x)
        x = x.permute(1, 0, 2)
        return x, lstm_hidden


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_size,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, num_actions)

        self.num_actions = num_actions

    def forward(self, state, softmax_dim=-1):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        # x = F.tanh(self.linear4(x))

        probs = F.softmax(self.output(x), dim=softmax_dim)

        return probs

    def evaluate(self, state, epsilon=1e-8):
        """
        generate sampled action with state as input wrt the policy network;
        """
        probs = self.forward(state, softmax_dim=-1)
        log_probs = torch.log(probs)

        # Avoid numerical instability. Ref: https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/model.py#L98
        z = (probs == 0.0).float() * epsilon
        log_probs = torch.log(probs + z)

        return log_probs

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        dist = Categorical(probs)

        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action


class PolicyNetworkLSTM(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_size,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
    ):
        super(PolicyNetworkLSTM, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_actions)

        self.num_actions = num_actions

    def forward(self, state, hidden_in, softmax_dim=-1):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1, 0, 2)
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        # x = F.tanh(self.linear4(x))
        x, lstm_hidden = self.lstm1(x, hidden_in)
        x = F.tanh(self.linear3(x))

        x = x.permute(1, 0, 2)  # permute back

        probs = F.softmax(self.output(x), dim=softmax_dim)

        return probs, lstm_hidden

    def evaluate(self, state, hidden_in, epsilon=1e-8):
        """
        generate sampled action with state as input wrt the policy network;
        """
        probs, hidden_out = self.forward(state, hidden_in, softmax_dim=-1)
        log_probs = torch.log(probs)

        # Avoid numerical instability. Ref: https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/model.py#L98
        z = (probs == 0.0).float() * epsilon
        log_probs = torch.log(probs + z)

        return log_probs, hidden_out

    def get_action(self, state, hidden_in, deterministic):
        state = (
            torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        )  # TODO maybe need to adjust unsqueeze here
        probs, hidden_out = self.forward(state, hidden_in)
        dist = Categorical(probs)

        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action, hidden_out


class SAC_Trainer:
    def __init__(self, replay_buffer, hidden_dim):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(
            device
        )
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(
            device
        )
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.log_alpha = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device=device
        )
        print("Soft Q Network (1,2): ", self.soft_q_net1)
        print("Policy Network: ", self.policy_net)

        for target_param, param in zip(
            self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()
        ):
            target_param.data.copy_(param.data)
        for target_param, param in zip(
            self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()
        ):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(
        self,
        batch_size,
        reward_scale=10.0,
        auto_entropy=True,
        target_entropy=-2,
        gamma=0.99,
        soft_tau=1e-2,
    ):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = (
            self.replay_buffer.sample(batch_size)
        )
        # print('sample:', state, action,  reward, done)

        # action = torch.Tensor(action).to(torch.int64).to(device)
        action = ints_to_tensor(action).to(device)
        # print(action.shape)
        # state = torch.FloatTensor(state).to(device)
        state = floats_to_tensor(state).to(device)
        # next_state = torch.FloatTensor(next_state).to(device)
        next_state = floats_to_tensor(next_state).to(device)
        # reward = ( torch.FloatTensor(reward).unsqueeze(1).to(device))  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = (
            floats_to_tensor(reward).unsqueeze(-1).to(device)
        )  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        # done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        done = floats_to_tensor(done).unsqueeze(-1).to(device)
        predicted_q_value1 = self.soft_q_net1(state)
        print(action.shape)
        print(action.unsqueeze(-1).shape)
        print(predicted_q_value1.shape)
        print(reward)
        exit(0)
        predicted_q_value1 = predicted_q_value1.gather(1, action.unsqueeze(-1))
        predicted_q_value2 = self.soft_q_net2(state)
        predicted_q_value2 = predicted_q_value2.gather(1, action.unsqueeze(-1))
        log_prob = self.policy_net.evaluate(state)
        with torch.no_grad():
            next_log_prob = self.policy_net.evaluate(next_state)

        # reward = (
        #     reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)
        # )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        self.alpha = self.log_alpha.exp()
        target_q_min = next_log_prob.exp() * (
            torch.min(
                self.target_soft_q_net1(next_state),
                self.target_soft_q_net2(next_state),
            )
            - self.alpha * next_log_prob
        ).sum(dim=-1).unsqueeze(-1)

        target_q_value = (
            reward + (1 - done) * gamma * target_q_min
        )  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(
            predicted_q_value1, target_q_value.detach()
        )  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(
            predicted_q_value2, target_q_value.detach()
        )

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        with torch.no_grad():
            predicted_new_q_value = torch.min(
                self.soft_q_net1(state),
                self.soft_q_net2(state),
            )
        policy_loss = (
            (log_prob.exp() * (self.alpha * log_prob - predicted_new_q_value))
            .sum(dim=-1)
            .mean()
        )

        print(policy_loss.shape)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            self.alpha = 1.0
            alpha_loss = 0

        print("q loss: ", q_value_loss1.item(), q_value_loss2.item())
        print("policy loss: ", policy_loss.item())

        # Soft update the target value net
        for target_param, param in zip(
            self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()
        ):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(
            self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()
        ):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + "_q1")
        torch.save(self.soft_q_net2.state_dict(), path + "_q2")
        torch.save(self.policy_net.state_dict(), path + "_policy")

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + "_q1"))
        self.soft_q_net2.load_state_dict(torch.load(path + "_q2"))
        self.policy_net.load_state_dict(torch.load(path + "_policy"))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig("sac_v2.png")
    # plt.show()


replay_buffer_size = 1e6
# replay_buffer = ReplayBuffer(replay_buffer_size)
replay_buffer = ReplayBufferLSTM2(replay_buffer_size)

# choose env
env = gym.make("CartPole-v1")
# env = gym.make("MountainCar-v0")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n  # discrete

# hyper-parameters for RL training
max_episodes = 10000
max_steps = 200
frame_idx = 0
# batch_size = 256
# batch_size = 20
batch_size = 2
update_itr = 1
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim = 64
rewards = []
model_path = "./model/sac_discrete_v2"
target_entropy = -1.0 * action_dim
# target_entropy = 0.98 * -np.log(1 / action_dim)

sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim)

if __name__ == "__main__":
    if args.train:
        # training loop
        for eps in range(max_episodes):
            state = env.reset()
            # episode_reward = 0

            last_action = env.action_space.sample()
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []
            hidden_out = (
                torch.zeros([1, 1, hidden_dim], dtype=torch.float, device=device),
                torch.zeros([1, 1, hidden_dim], dtype=torch.float, device=device),
            )  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

            for step in range(max_steps):
                hidden_in = hidden_out
                action = sac_trainer.policy_net.get_action(
                    state, deterministic=DETERMINISTIC
                )
                next_state, reward, done, _ = env.step(action)
                # env.render()

                # replay_buffer.push(state, action, reward, next_state, done)
                if step == 0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out
                episode_state.append(state)
                episode_action.append(action.item())
                episode_last_action.append(last_action)
                episode_reward.append(reward)
                episode_next_state.append(next_state)
                episode_done.append(done)

                state = next_state
                # episode_reward += reward
                last_action = action
                frame_idx += 1

                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):
                        _ = sac_trainer.update(
                            batch_size,
                            reward_scale=10.0,
                            auto_entropy=AUTO_ENTROPY,
                            target_entropy=target_entropy,
                        )

                if done:
                    break

            replay_buffer.push(
                ini_hidden_in,
                ini_hidden_out,
                episode_state,
                episode_action,
                episode_last_action,
                episode_reward,
                episode_next_state,
                episode_done,
            )

            # if eps % 20 == 0 and eps > 0:  # plot and model saving interval
            # plot(rewards)
            # np.save("rewards", rewards)
            # sac_trainer.save_model(model_path)
            print(
                "Episode: ",
                eps,
                "| Episode Reward: ",
                np.sum(episode_reward),
                "| Episode Length: ",
                step,
            )
            rewards.append(episode_reward)
        sac_trainer.save_model(model_path)

    if args.test:
        sac_trainer.load_model(model_path)
        for eps in range(10):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = sac_trainer.policy_net.get_action(
                    state, deterministic=DETERMINISTIC
                )
                next_state, reward, done, _ = env.step(action)
                env.render()

                episode_reward += reward
                state = next_state

            print("Episode: ", eps, "| Episode Reward: ", episode_reward)
