from enum import Enum, auto

import torch
import torch.nn as nn
from rl_utils.common import make_mlp_layers


class RewardInputType(Enum):
    ACTION = auto()
    NEXT_STATE = auto()
    CUR_NEXT_STATE = auto()


class NeuralReward(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        reward_hidden_dim,
        n_hidden_layers,
        cost_take_dim=-1,
        include_tanh=False,
        reward_type=None,
        clamp_max=None,
    ):
        super().__init__()
        if reward_type is None:
            self.reward_type = RewardInputType.NEXT_STATE
        else:
            self.reward_type = RewardInputType[reward_type]
        self.cost_take_dim = cost_take_dim

        obs_size = obs_shape[0] if cost_take_dim == -1 else cost_take_dim

        if self.reward_type == RewardInputType.ACTION:
            input_size = obs_size + action_dim
        elif self.reward_type == RewardInputType.NEXT_STATE:
            input_size = obs_size
        elif self.reward_type == RewardInputType.CUR_NEXT_STATE:
            input_size = obs_size + obs_size

        net_layers = make_mlp_layers(input_size, 1, reward_hidden_dim, n_hidden_layers)
        if include_tanh:
            net_layers.append(nn.Tanh())
        self.net = nn.Sequential(*net_layers)
        self.clamp_max = clamp_max

    def forward(self, cur_obs=None, actions=None, next_obs=None):
        if self.cost_take_dim != -1:
            if cur_obs is not None:
                cur_obs = cur_obs[:, :, : self.cost_take_dim]
            if next_obs is not None:
                next_obs = next_obs[:, :, : self.cost_take_dim]

        if self.reward_type == RewardInputType.ACTION:
            inputs = [cur_obs, actions]
        elif self.reward_type == RewardInputType.NEXT_STATE:
            inputs = [next_obs]
        elif self.reward_type == RewardInputType.CUR_NEXT_STATE:
            inputs = [cur_obs, next_obs]
        else:
            raise ValueError()
        inputs = torch.cat(inputs, dim=-1)

        ret = self.net(inputs)
        if self.clamp_max is not None:
            ret = torch.clamp(ret, min=-self.clamp_max, max=self.clamp_max)
        return ret
