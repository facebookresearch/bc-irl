# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from rl_utils.common import DictDataset, make_mlp_layers
from torch.utils.data import DataLoader
from torchrl.trainers import BatchSubSampler

from imitation_learning.common.utils import log_finished_rewards


class NeuralReward(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        reward_hidden_dim,
        cost_take_dim,
        n_hidden_layers,
    ):
        super().__init__()
        self.cost_take_dim = cost_take_dim

        obs_size = obs_shape[0] if cost_take_dim == -1 else cost_take_dim
        self.net = nn.Sequential(
            *make_mlp_layers(obs_size, 1, reward_hidden_dim, n_hidden_layers)
        )

    def forward(self, obs):
        return self.net(obs)


class GCL(nn.Module):
    def __init__(
        self,
        reward: DictConfig,
        reward_opt: DictConfig,
        get_dataset_fn,
        batch_size: int,
        device,
        policy_updater: DictConfig,
        should_update_reward: bool,
        policy,
        num_envs,
        **kwargs
    ):
        super().__init__()
        self.reward = instantiate(reward).to(device)
        self.policy_updater = instantiate(policy_updater, policy=policy)
        self.batch_size = batch_size

        self.dataset = call(get_dataset_fn)
        self.expert_data = DataLoader(self.dataset, batch_size, shuffle=True)

        self.reward_opt = instantiate(reward_opt, params=self.reward.parameters())
        self._ep_rewards = torch.zeros(num_envs, device=device)
        self.should_update_reward = should_update_reward

    def state_dict(self, **kwargs):
        return {
            **super().state_dict(**kwargs),
            "reward_opt": self.reward_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, should_load_opt):
        opt_state = state_dict.pop("reward_opt")
        if should_load_opt:
            self.reward_opt.load_state_dict(opt_state)
        return super().load_state_dict(state_dict)

    def viz_reward(self, cur_obs=None, action=None, next_obs=None) -> torch.Tensor:
        return self.reward(next_obs)

    def update(self, policy, rollouts, logger, **kwargs):
        if self.should_update_reward:

            reward_samples = []

            data_gen = BatchSubSampler(batch_size=self.batch_size)
            flat_rollouts = rollouts.view(-1)
            for expert_batch in self.expert_data:
                agent_batch = data_gen(flat_rollouts)
                ac_eval = policy.evaluate_actions(agent_batch)

                reward_demos = self.reward(expert_batch["next_observations"])
                reward_samples = self.reward(agent_batch["next_observation"])

                loss_IOC = -(
                    torch.mean(reward_demos)
                    - (
                        torch.logsumexp(
                            reward_samples - ac_eval["log_prob"].view(-1, 1),
                            dim=0,
                            keepdim=True,
                        )
                        - torch.log(torch.Tensor([len(reward_samples)]))
                    )
                )
                self.reward_opt.zero_grad()
                loss_IOC.backward()
                self.reward_opt.step()
                logger.collect_info("irl_loss", loss_IOC.item())

        with torch.no_grad():
            rollouts["reward"] = self.reward(rollouts["next_observation"])
            self._ep_rewards = log_finished_rewards(rollouts, self._ep_rewards, logger)
        self.policy_updater.update(policy, rollouts, logger)
