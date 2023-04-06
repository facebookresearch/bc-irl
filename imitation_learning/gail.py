# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.trainers import BatchSubSampler

import imitation_learning
from imitation_learning.common.utils import log_finished_rewards


class GAIL(nn.Module):
    def __init__(
        self,
        discriminator: DictConfig,
        policy_updater: DictConfig,
        get_dataset_fn,
        batch_size: int,
        num_discrim_batches: int,
        discrim_opt: DictConfig,
        reward_update_freq: int,
        device,
        policy,
        num_envs,
        **kwargs,
    ):
        super().__init__()

        self.discriminator = instantiate(discriminator).to(device)
        self.policy_updater = instantiate(policy_updater, policy=policy)

        self.dataset = call(get_dataset_fn)
        self.expert_data = DataLoader(self.dataset, batch_size, shuffle=True)
        self.discrim_opt = instantiate(
            discrim_opt, params=self.discriminator.parameters()
        )
        self.reward_update_freq = reward_update_freq
        self._n_updates = 0
        self.batch_size = batch_size
        self.num_discrim_batches = num_discrim_batches

        self.device = device
        self._ep_rewards = torch.zeros(num_envs, device=self.device)

    def state_dict(self, **kwargs):
        return {
            **super().state_dict(**kwargs),
            "discrim_opt": self.discrim_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, should_load_opt=False):
        opt_state = state_dict.pop("discrim_opt")
        return super().load_state_dict(state_dict)

    def viz_reward(self, cur_obs=None, action=None, next_obs=None) -> torch.Tensor:
        return self.discriminator.get_reward(
            TensorDict(
                {
                    "observation": cur_obs,
                    "next_observation": next_obs,
                },
                batch_size=[cur_obs.shape[0]],
            ),
            viz_reward=True,
        )

    def _update_discriminator(self, policy, rollouts, logger):
        num_batches = len(rollouts) // self.batch_size
        cur_num_batches = 0
        data_gen = BatchSubSampler(batch_size=self.batch_size)
        flat_rollouts = rollouts.view(-1)

        def convert_expert_key(expert_batch):
            renames = {
                "observations": "observation",
                "next_observations": "next_observation",
                "actions": "action",
                "terminals": "done",
            }
            return TensorDict(
                {renames.get(k, k): v for k, v in expert_batch.items()},
                batch_size=[expert_batch["actions"].shape[0]],
            )

        for expert_batch in self.expert_data:
            agent_batch = data_gen(flat_rollouts)
            if (
                self.num_discrim_batches != -1
                and self.num_discrim_batches <= cur_num_batches
            ):
                break
            agent_d = self.discriminator(
                agent_batch,
                policy=policy,
            )
            expert_d = self.discriminator(
                convert_expert_key(expert_batch),
                policy=policy,
            )

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d, torch.ones_like(expert_d, device=self.device)
            )
            agent_loss = F.binary_cross_entropy_with_logits(
                agent_d, torch.zeros_like(agent_d, device=self.device)
            )

            loss = expert_loss + agent_loss

            self.discrim_opt.zero_grad()
            loss.backward()
            self.discrim_opt.step()

            logger.collect_info("expert_loss", expert_loss.item())
            logger.collect_info("agent_loss", agent_loss.item())
            logger.collect_info("discim_loss", loss.item())
            cur_num_batches += 1

    def update(self, policy, rollouts, logger, **kwargs):
        if (
            self.reward_update_freq != -1
            and self._n_updates % self.reward_update_freq == 0
        ):
            self._update_discriminator(policy, rollouts, logger)

        with torch.no_grad():
            rollouts["rewards"] = self.discriminator.get_reward(
                rollouts,
                policy=policy,
            )
            self._ep_rewards = log_finished_rewards(rollouts, self._ep_rewards, logger)
        self.policy_updater.update(policy, rollouts, logger)
        self._n_updates += 1
