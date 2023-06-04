# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, auto

import torch
import torch.nn as nn
from hydra.utils import instantiate
from rl_utils.common import make_mlp_layers


def full_reset_init(old_policy, policy_cfg, **kwargs):
    return instantiate(policy_cfg)


def reg_init(old_policy, **kwargs):
    return old_policy


class StructuredReward(nn.Module):
    def __init__(self, obs_shape, **kwargs):
        super().__init__()
        self.center = nn.Parameter(torch.randn(obs_shape[0]))

    def forward(self, X):
        return -1.0 * ((X - self.center) ** 2).mean(-1, keepdims=True)

    def log(self, logger):
        for i, center_val in enumerate(self.center):
            logger.collect_info(f"reward_{i}", center_val.item())


class GtReward(nn.Module):
    def __init__(
        self,
    ):
        pass

    def forward(self, cur_obs=None, actions=None, next_obs=None):
        cur_dist = torch.linalg.norm(cur_obs, dim=-1)
        reward = torch.full(cur_dist.shape, -self._slack)
        assign = -self._slack * cur_dist
        should_give_reward = cur_dist < self._reward_thresh
        reward[should_give_reward] = assign[should_give_reward]
        return reward
