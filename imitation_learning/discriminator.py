# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, auto
from typing import Tuple

import torch
import torch.nn as nn
from rl_utils.common import make_mlp_layers


class GailRewardType(Enum):
    AIRL = auto()
    GAIL = auto()
    RAW = auto()


class GailDiscriminator(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int],
        action_dim: int,
        reward_hidden_dim: int,
        n_hidden_layers: int,
        cost_take_dim: int,
        use_actions: bool,
        reward_type: str,
    ):
        super().__init__()
        self.cost_take_dim = cost_take_dim
        input_size = obs_shape[0] if cost_take_dim == -1 else cost_take_dim
        if use_actions:
            input_size += action_dim
        self.action_input = use_actions

        self.discrim_net = nn.Sequential(
            *make_mlp_layers(input_size, 1, reward_hidden_dim, n_hidden_layers)
        )

        self.reward_type = GailRewardType[reward_type]

    def forward(self, cur_obs=None, actions=None, **kwargs):
        if self.cost_take_dim != -1:
            if cur_obs is not None:
                cur_obs = cur_obs[:, :, : self.cost_take_dim]

        if self.action_input:
            inputs = torch.cat([cur_obs, actions], -1)
        else:
            inputs = cur_obs

        return self.discrim_net(inputs)

    def get_reward(self, cur_obs=None, actions=None, **kwargs):
        d_val = self.forward(cur_obs, actions)
        s = torch.sigmoid(d_val)
        eps = 1e-20
        if self.reward_type == GailRewardType.AIRL:
            reward = (s + eps).log() - (1 - s + eps).log()
        elif self.reward_type == GailRewardType.GAIL:
            reward = (s + eps).log()
        elif self.reward_type == GailRewardType.RAW:
            reward = d_val

        return reward


class AirlDiscriminator(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int],
        action_dim: int,
        reward_hidden_dim: int,
        cost_take_dim: int,
        n_hidden_layers: int,
        use_shaped_reward: bool,
        gamma: float,
        airl_reward_bonus: float,
    ):
        super().__init__()
        self.cost_take_dim = cost_take_dim
        state_size = obs_shape[0] if cost_take_dim == -1 else cost_take_dim

        self.g = nn.Sequential(
            *make_mlp_layers(state_size, 1, reward_hidden_dim, n_hidden_layers)
        )
        self.h = nn.Sequential(
            *make_mlp_layers(state_size, 1, reward_hidden_dim, n_hidden_layers)
        )

        self.use_shaped_reward = use_shaped_reward
        self.gamma = gamma
        self.airl_reward_bonus = airl_reward_bonus

    def f(self, td, force_no_shaped=False, **kwargs):
        rs = self.g(td["observation"])
        if self.use_shaped_reward and not force_no_shaped:
            vs = self.h(td["observation"])
            next_vs = self.h(td["next_observation"])
            masks = (~td["done"]).float()
            return rs + (self.gamma * masks * next_vs) - vs
        else:
            return rs

    def forward(self, td, policy):
        log_p = self.f(td)

        with torch.no_grad():
            log_q = policy.evaluate_actions(td)["log_prob"].view(-1, 1)

        return log_p - log_q

    def get_reward(self, td, policy=None, viz_reward=False, **kwargs):
        log_p = self.f(td, force_no_shaped=viz_reward)
        if viz_reward:
            return log_p

        with torch.no_grad():
            log_q = policy.evaluate_actions(td)["log_prob"].unsqueeze(-1)

        logits = log_p - (self.airl_reward_bonus * log_q)
        s = torch.sigmoid(logits)
        eps = 1e-20
        return (s + eps).log() - (1 - s + eps).log()
