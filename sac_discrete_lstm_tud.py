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
import copy
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
from buffer import *
from foldedtensor import as_folded_tensor
from torch.distributions.normal import Normal

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


class LSTM_GaussianActor(nn.Module):
    """Defines recurrent, stochastic actor based on a Gaussian distribution."""

    def __init__(
        self, action_dim, state_shape, use_past_actions, log_std_min=-20, log_std_max=2
    ):
        super(LSTM_GaussianActor, self).__init__()

        self.use_past_actions = use_past_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # current feature extraction
        self.curr_fe_dense1 = nn.Linear(state_shape, 128)
        self.curr_fe_dense2 = nn.Linear(128, 128)

        # memory
        if use_past_actions:
            self.mem_dense = nn.Linear(state_shape + action_dim, 128)
        else:
            self.mem_dense = nn.Linear(state_shape, 128)
        self.mem_LSTM = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=1, batch_first=True
        )

        # post combination
        self.post_comb_dense = nn.Linear(128 + 128, 128)

        # output mu and log_std
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, s, s_hist, a_hist, hist_len, deterministic, with_logprob):
        """Returns action and it's logprob for given obs and history. o, o_hist, a_hist, hist_len are torch tensors. Args:

        s:        torch.Size([batch_size, state_shape])
        s_hist:   torch.Size([batch_size, history_length, state_shape])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)

        deterministic: bool (whether to use mean as a sample, only at test time)
        with_logprob:  bool (whether to return logprob of sampled action as well, else second tuple element below will be 'None')

        returns:       (torch.Size([batch_size, action_dim]), torch.Size([batch_size, action_dim]), act_net_info (dict))

        Note:
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, state_shape)

        The call <out, (hidden, cell) = LSTM(x)> results in:
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        # ------ current feature extraction ------
        curr_fe = F.relu(self.curr_fe_dense1(s))
        curr_fe = F.relu(self.curr_fe_dense2(curr_fe))

        # ------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem = F.relu(self.mem_dense(torch.cat([s_hist, a_hist], dim=2)))
        else:
            x_mem = F.relu(self.mem_dense(s_hist))

        # LSTM
        # self.mem_LSTM.flatten_parameters()
        extracted_mem, (_, _) = self.mem_LSTM(x_mem)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1

        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem = extracted_mem[torch.arange(extracted_mem.size(0)), h_idx]

        # mask no-history cases to yield zero extracted memory
        hidden_mem[hist_len == 0] = 0.0

        # ------ post combination ------
        # concate current feature extraction with generated memory
        x = torch.cat([curr_fe, hidden_mem], dim=1)

        # final dense layer
        x = F.relu(self.post_comb_dense(x))

        # compute mean, log_std and std of Gaussian
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # ------ having mu and std, compute actions and log_probs -------
        # construct pre-squashed distribution
        pi_distribution = Normal(mu, std)

        # sample action, deterministic only used for evaluating policy at test time
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        # compute logprob from Gaussian and then correct it for the Tanh squashing
        if with_logprob:
            # this does not exactly match the expression given in Appendix C in the paper, but it is
            # equivalent and according to SpinningUp OpenAI numerically much more stable
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )

            # logp_pi sums in both prior steps over all actions,
            # since these are assumed to be independent Gaussians and can thus be factorized into their margins
            # however, shape is now torch.Size([batch_size]), but we want torch.Size([batch_size, 1])
            logp_pi = logp_pi.reshape((-1, 1))

        else:
            logp_pi = None

        # squash action to [-1, 1]
        pi_action = torch.tanh(pi_action)

        # ------ return ---------
        # create dict for logging
        act_net_info = dict(
            Actor_CurFE=curr_fe.detach().mean().cpu().numpy(),
            Actor_ExtMemory=hidden_mem.detach().mean().cpu().numpy(),
        )

        # return squashed action, it's logprob and logging info
        return pi_action, logp_pi, act_net_info


class LSTM_Actor(nn.Module):
    """Defines recurrent deterministic actor."""

    def __init__(self, action_dim, state_shape, use_past_actions) -> None:
        super(LSTM_Actor, self).__init__()

        self.use_past_actions = use_past_actions

        # current feature extraction
        self.curr_fe_dense1 = nn.Linear(state_shape, 128)
        self.curr_fe_dense2 = nn.Linear(128, 128)

        # memory
        if use_past_actions:
            self.mem_dense = nn.Linear(state_shape + action_dim, 128)
        else:
            self.mem_dense = nn.Linear(state_shape, 128)
        self.mem_LSTM = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=1, batch_first=True
        )

        # post combination
        self.post_comb_dense1 = nn.Linear(128 + 128, 128)
        self.post_comb_dense2 = nn.Linear(128, action_dim)

    def forward(self, s, s_hist, a_hist, hist_len) -> tuple:
        """s, s_hist, hist_len are torch tensors. Shapes:
        s:        torch.Size([batch_size, state_shape])
        s_hist:   torch.Size([batch_size, history_length, state_shape])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)

        returns: output with shape torch.Size([batch_size, action_dim]), act_net_info (dict)

        Note:
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, state_shape)

        The call <out, (hidden, cell) = LSTM(x)> results in:
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        # ------ current feature extraction ------
        curr_fe = F.relu(self.curr_fe_dense1(s))
        curr_fe = F.relu(self.curr_fe_dense2(curr_fe))

        # ------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem = F.relu(self.mem_dense(torch.cat([s_hist, a_hist], dim=2)))
        else:
            x_mem = F.relu(self.mem_dense(s_hist))

        # LSTM
        # self.mem_LSTM.flatten_parameters()
        extracted_mem, (_, _) = self.mem_LSTM(x_mem)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1

        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem = extracted_mem[torch.arange(extracted_mem.size(0)), h_idx]

        # mask no-history cases to yield zero extracted memory
        hidden_mem[hist_len == 0] = 0.0

        # ------ post combination ------
        # concate current feature extraction with generated memory
        x = torch.cat([curr_fe, hidden_mem], dim=1)

        # final dense layers
        x = F.relu(self.post_comb_dense1(x))
        x = torch.tanh(self.post_comb_dense2(x))

        # create dict for logging
        act_net_info = dict(
            Actor_CurFE=curr_fe.detach().mean().cpu().numpy(),
            Actor_ExtMemory=hidden_mem.detach().mean().cpu().numpy(),
        )

        # return output
        return x, act_net_info


class LSTM_Critic(nn.Module):
    """Defines recurrent critic network to compute Q-values."""

    def __init__(self, action_dim, state_shape, use_past_actions) -> None:
        super(LSTM_Critic, self).__init__()

        self.use_past_actions = use_past_actions

        # current feature extraction
        self.curr_fe_dense1 = nn.Linear(state_shape + action_dim, 128)
        self.curr_fe_dense2 = nn.Linear(128, 128)

        # memory
        if use_past_actions:
            self.mem_dense = nn.Linear(state_shape + action_dim, 128)
        else:
            self.mem_dense = nn.Linear(state_shape, 128)
        self.mem_LSTM = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=1, batch_first=True
        )

        # post combination
        self.post_comb_dense1 = nn.Linear(128 + 128, 128)
        self.post_comb_dense2 = nn.Linear(128, 1)

    def forward(self, s, a, s_hist, a_hist, hist_len, log_info=True) -> tuple:
        """s, s_hist, a_hist are torch tensors. Shapes:
        s:        torch.Size([batch_size, state_shape])
        a:        torch.Size([batch_size, action_dim])
        s_hist:   torch.Size([batch_size, history_length, state_shape])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)
        log_info: Bool, whether to return logging dict

        returns: output with shape torch.Size([batch_size, 1]), critic_net_info (dict) (if log_info)

        Note:
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, state_shape)

        The call <out, (hidden, cell) = LSTM(x)> results in:
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        # ------ current feature extraction ------
        # concatenate obs and act
        sa = torch.cat([s, a], dim=1)
        curr_fe = F.relu(self.curr_fe_dense1(sa))
        curr_fe = F.relu(self.curr_fe_dense2(curr_fe))

        # ------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem = F.relu(self.mem_dense(torch.cat([s_hist, a_hist], dim=2)))
        else:
            x_mem = F.relu(self.mem_dense(s_hist))

        # LSTM
        # self.mem_LSTM.flatten_parameters()
        extracted_mem, (_, _) = self.mem_LSTM(x_mem)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1

        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem = extracted_mem[torch.arange(extracted_mem.size(0)), h_idx]

        # mask no-history cases to yield zero extracted memory
        hidden_mem[hist_len == 0] = 0.0

        # ------ post combination ------
        # concatenate current feature extraction with generated memory
        x = torch.cat([curr_fe, hidden_mem], dim=1)

        # final dense layers
        x = F.relu(self.post_comb_dense1(x))
        x = self.post_comb_dense2(x)

        # create dict for logging
        if log_info:
            critic_net_info = dict(
                Critic_CurFE=curr_fe.detach().mean().cpu().numpy(),
                Critic_ExtMemory=hidden_mem.detach().mean().cpu().numpy(),
            )
            return x, critic_net_info
        else:
            return x


class LSTM_Double_Critic(nn.Module):
    def __init__(self, action_dim, state_shape, use_past_actions) -> None:
        super(LSTM_Double_Critic, self).__init__()

        self.LSTM_Q1 = LSTM_Critic(
            action_dim=action_dim,
            state_shape=state_shape,
            use_past_actions=use_past_actions,
        )

        self.LSTM_Q2 = LSTM_Critic(
            action_dim=action_dim,
            state_shape=state_shape,
            use_past_actions=use_past_actions,
        )

    def forward(self, s, a, s_hist, a_hist, hist_len) -> tuple:
        q1 = self.LSTM_Q1(s, a, s_hist, a_hist, hist_len, log_info=False)
        q2, critic_net_info = self.LSTM_Q2(
            s, a, s_hist, a_hist, hist_len, log_info=True
        )

        return q1, q2, critic_net_info

    def single_forward(self, s, a, s_hist, a_hist, hist_len):
        q1 = self.LSTM_Q1(s, a, s_hist, a_hist, hist_len, log_info=False)

        return q1


class LSTMSACAgent:
    def __init__(self):
        # super().__init__(c)

        # attributes and hyperparameters
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.tau = 0.001
        # self.critic_weights = c.critic_weights
        # self.net_struc_actor = c.net_struc_actor
        # self.net_struc_critic = c.net_struc_critic

        self.device = device
        self.num_actions = action_dim
        self.state_shape = state_dim
        self.buffer_length = int(1e6)
        self.batch_size = 32
        self.state_type = "feature"
        self.loss = "MSELoss"
        self.gamma = 0.99
        self.grad_rescale = False
        self.grad_clip = False

        self.lr_temp = 3e-4
        self.temp_tuning = True
        # self.init_temp = getattr(c.Agent, agent_name)["init_temp"]

        self.needs_history = True
        self.history_length = 2
        self.use_past_actions = False

        # dynamic or static temperature
        # define target entropy
        self.target_entropy = -self.num_actions

        # optimize log(temperature) instead of temperature
        self.log_temperature = torch.zeros(1, requires_grad=True, device=self.device)

        # define temperature optimizer
        self.temp_optimizer = optim.Adam([self.log_temperature], lr=self.lr_temp)

        # replay buffer
        self.replay_buffer = UniformReplayBuffer_LSTM(
            state_type=self.state_type,
            state_shape=self.state_shape,
            buffer_length=self.buffer_length,
            batch_size=self.batch_size,
            device=self.device,
            disc_actions=False,
            action_dim=self.num_actions,
            history_length=self.history_length,
        )

        # init actor and critic
        # self.actor = LSTM_GaussianActor(
        self.actor = LSTM_GaussianActor(
            state_shape=self.state_shape,
            action_dim=self.num_actions,
            use_past_actions=self.use_past_actions,
        ).to(self.device)

        self.critic = LSTM_Double_Critic(
            state_shape=self.state_shape,
            action_dim=self.num_actions,
            use_past_actions=self.use_past_actions,
        ).to(self.device)

        # init target net
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # define optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    @torch.no_grad()
    def select_action(self, s, s_hist, a_hist, hist_len):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        s:        np.array with shape (state_shape,)
        s_hist:   np.array with shape (history_length, state_shape)
        a_hist:   np.array with shape (history_length, action_dim)
        hist_len: int

        returns: np.array with shape (action_dim,)
        """
        # reshape arguments and convert to tensors
        s = (
            torch.tensor(s, dtype=torch.float32)
            .view(1, self.state_shape)
            .to(self.device)
        )
        s_hist = (
            torch.tensor(s_hist, dtype=torch.float32)
            .view(1, self.history_length, self.state_shape)
            .to(self.device)
        )
        a_hist = (
            torch.tensor(a_hist, dtype=torch.float32)
            .view(1, self.history_length, self.num_actions)
            .to(self.device)
        )
        hist_len = torch.tensor(hist_len).to(self.device)

        # forward pass
        a, _, _ = self.actor(
            s,
            s_hist,
            a_hist,
            hist_len,
            deterministic=True,
            with_logprob=False,
        )

        # reshape actions
        return a.cpu().numpy().reshape(self.num_actions)

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)

    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
        with torch.no_grad():
            # target actions come from current policy (no target actor)
            target_a, target_logp_a, _ = self.actor(
                s=s2,
                s_hist=s2_hist,
                a_hist=a2_hist,
                hist_len=hist_len2,
                deterministic=False,
                with_logprob=True,
            )

            # Q-value of next state-action pair
            Q_next1, Q_next2, _ = self.target_critic(
                s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2
            )
            Q_next = torch.min(Q_next1, Q_next2)

            # target
            y = r + self.gamma * (1 - d) * (Q_next - self.temperature * target_logp_a)
        return y

    def _compute_loss(self, Q, y, reduction="mean"):
        if self.loss == "MSELoss":
            return F.mse_loss(Q, y, reduction=reduction)

        elif self.loss == "SmoothL1Loss":
            return F.smooth_l1_loss(Q, y, reduction=reduction)

    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""
        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        s_hist, a_hist, hist_len, s2_hist, a2_hist, hist_len2, s, a, r, s2, d = batch

        # get current temperature
        if self.temp_tuning:
            self.temperature = torch.exp(self.log_temperature).detach()

        # -------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()

        # calculate current estimated Q-values
        Q1, Q2, critic_net_info = self.critic(
            s=s, a=a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len
        )

        # calculate targets
        y = self._compute_target(s2_hist, a2_hist, hist_len2, r, s2, d)

        # calculate loss
        critic_loss = self._compute_loss(Q1, y) + self._compute_loss(Q2, y)

        # compute gradients
        critic_loss.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.critic.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)

        # perform optimizing step
        self.critic_optimizer.step()

        # log critic training
        # self.logger.store(
        #     Critic_loss=critic_loss.detach().cpu().numpy().item(), **critic_net_info
        # )
        # self.logger.store(Q_val=Q1.detach().mean().cpu().numpy().item())

        # -------- train actor --------
        # freeze critic so no gradient computations are wasted while training actor
        for param in self.critic.parameters():
            param.requires_grad = False

        # clear gradients
        self.actor_optimizer.zero_grad()

        # get current actions via actor
        curr_a, curr_a_logprob, act_net_info = self.actor(
            s=s,
            s_hist=s_hist,
            a_hist=a_hist,
            hist_len=hist_len,
            deterministic=False,
            with_logprob=True,
        )

        # compute Q1, Q2 values for current state and actor's actions
        Q1_curr_a, Q2_curr_a, _ = self.critic(
            s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len
        )
        Q_curr_a = torch.min(Q1_curr_a, Q2_curr_a)

        # compute policy loss (which is based on min Q1, Q2 instead of just Q1 as in TD3, plus consider entropy regularization)
        actor_loss = (self.temperature * curr_a_logprob - Q_curr_a).mean()

        # compute gradients
        actor_loss.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.actor.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)

        # perform step with optimizer
        self.actor_optimizer.step()

        # unfreeze critic so it can be trained in next iteration
        for param in self.critic.parameters():
            param.requires_grad = True

        # log actor training
        # self.logger.store(
        #     Actor_loss=actor_loss.detach().cpu().numpy().item(), **act_net_info
        # )

        # ------- update temperature --------
        if self.temp_tuning:
            # clear gradients
            self.temp_optimizer.zero_grad()

            # calculate loss
            temperature_loss = (
                -self.log_temperature
                * (curr_a_logprob + self.target_entropy).detach().mean()
            )

            # compute gradients
            temperature_loss.backward()

            # perform optimizer step
            self.temp_optimizer.step()

        # ------- Update target networks -------
        self.polyak_update()

    @torch.no_grad()
    def polyak_update(self):
        """Soft update of target network weights."""

        for target_p, main_p in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_p.data.copy_(self.tau * main_p.data + (1 - self.tau) * target_p.data)


# choose env
# env = gym.make("CartPole-v1")
env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n  # discrete
action_dim = env.action_space.shape[0]  # discrete

# hyper-parameters for RL training
max_episodes = 10000
max_steps = 250
frame_idx = 0
# batch_size = 2
batch_size = 256
# batch_size = 25
update_itr = 1
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim = 512
rewards = []
model_path = "./model/sac_discrete_v2"
target_entropy = -1.0 * action_dim
# target_entropy = 0.98 * -np.log(1 / action_dim)

agent = LSTMSACAgent()

start_step = 300
upd_every = 4

if __name__ == "__main__":
    if args.train:
        # LSTM: init history
        s_hist = np.zeros((agent.history_length, agent.state_shape))
        a_hist = np.zeros((agent.history_length, agent.num_actions))
        hist_len = 0

        # training loop
        for eps in range(max_episodes):
            s = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                a = agent.select_action(
                    s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len
                )
                s2, r, d, __ = env.step(a)
                # env.render()

                episode_reward += r

                agent.memorize(s, a, r, s2, d)

                # LSTM: update history
                if hist_len == agent.history_length:
                    s_hist = np.roll(s_hist, shift=-1, axis=0)
                    s_hist[agent.history_length - 1, :] = s

                    a_hist = np.roll(a_hist, shift=-1, axis=0)
                    a_hist[agent.history_length - 1, :] = a
                else:
                    s_hist[hist_len] = s
                    a_hist[hist_len] = a
                    hist_len += 1
                    s = s2

                # train
                if (step >= start_step) and (step % upd_every == 0):
                    agent.train()

                s = s2

                if d:
                    # LSTM: reset history
                    s_hist = np.zeros((agent.history_length, agent.state_shape))
                    a_hist = np.zeros((agent.history_length, agent.num_actions))
                    hist_len = 0
                    break

            print(
                "Episode: ",
                eps,
                "| Episode Reward: ",
                episode_reward,
                "| Episode Length: ",
                step,
            )

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
