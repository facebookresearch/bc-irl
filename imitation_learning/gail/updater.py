# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader

from imitation_learning.common.plotting import plot_actions
from imitation_learning.common.utils import (create_next_obs,
                                             extract_transition_batch,
                                             log_finished_rewards)


def wass_grad_pen(
    expert_state,
    expert_action,
    policy_state,
    policy_action,
    use_actions,
    disc,
):
    num_dims = len(expert_state.shape) - 1
    alpha = torch.rand(expert_state.size(0), 1)
    alpha_state = (
        alpha.view(-1, *[1 for _ in range(num_dims)])
        .expand_as(expert_state)
        .to(expert_state.device)
    )
    mixup_data_state = alpha_state * expert_state + (1 - alpha_state) * policy_state
    mixup_data_state.requires_grad = True
    inputs = [mixup_data_state]

    if use_actions:
        alpha_action = alpha.expand_as(expert_action).to(expert_action.device)
        mixup_data_action = (
            alpha_action * expert_action + (1 - alpha_action) * policy_action
        )
        mixup_data_action.requires_grad = True
        inputs.append(mixup_data_action)
    else:
        mixup_data_action = []

    # disc = disc_fn(cur_obs=mixup_data_state, actions=mixup_data_action)
    disc = disc.g(mixup_data_state)
    ones = torch.ones(disc.size()).to(disc.device)

    grad = torch.autograd.grad(
        outputs=disc,
        inputs=inputs,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_pen = (grad.norm(2, dim=1) - 1).pow(2).mean()
    return grad_pen


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
        spectral_norm,
        grad_pen,
        **kwargs,
    ):
        super().__init__()
        self.discriminator = instantiate(discriminator).to(device)
        self.policy_updater = instantiate(policy_updater, policy=policy)
        self._grad_pen = grad_pen

        if spectral_norm:
            self._apply_spectral_norm()

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

    def _apply_spectral_norm(self):
        for name, module in self.discriminator.named_modules():
            # Only applies the spectral transformation to the high-level
            # modules and goes into the sequential modules and applies to
            # each element.
            if name == "" or "." in name:
                continue
            if isinstance(module, nn.Sequential):
                new_layers = []
                for i in range(len(module)):
                    layer = module[i]
                    if isinstance(layer, nn.Linear):
                        layer = spectral_norm(layer)
                    new_layers.append(layer)
                setattr(self.discriminator, name, nn.Sequential(*new_layers))
            elif isinstance(module, nn.Linear):
                setattr(self.discriminator, name, spectral_norm(module))

    def state_dict(self, **kwargs):
        return {
            **super().state_dict(**kwargs),
            "discrim_opt": self.discrim_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, should_load_opt):
        opt_state = state_dict.pop("discrim_opt")
        if should_load_opt:
            self.discrim_opt.load_state_dict(opt_state)
        return super().load_state_dict(state_dict)

    def viz_reward(self, cur_obs=None, action=None, next_obs=None) -> torch.Tensor:
        return self.discriminator.get_reward(
            cur_obs=cur_obs, actions=action, next_obs=next_obs, viz_reward=True
        )

    def _get_agent_samples(self, rollouts):
        num_batches = len(rollouts) // self.batch_size
        agent_data = rollouts.data_generator(num_batches, get_next_obs=True)
        if self.num_discrim_batches != -1:
            agent_data = itertools.islice(agent_data, self.num_discrim_batches)
        return agent_data

    def _update_discriminator(self, policy, rollouts, logger):
        num_batches = len(rollouts) // self.batch_size
        agent_data = self._get_agent_samples(rollouts)

        for expert_batch, agent_batch in zip(self.expert_data, agent_data):
            expert_d = self.discriminator(
                cur_obs=expert_batch["observations"],
                actions=expert_batch["actions"],
                next_obs=expert_batch["next_observations"],
                masks=(~expert_batch["terminals"].bool()).float(),
                policy=policy,
            )
            agent_d = self.discriminator(
                cur_obs=agent_batch["obs"],
                actions=agent_batch["action"],
                next_obs=agent_batch["next_obs"],
                masks=agent_batch["mask"],
                policy=policy,
            )

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d, torch.ones_like(expert_d, device=self.device)
            )
            agent_loss = F.binary_cross_entropy_with_logits(
                agent_d, torch.zeros_like(agent_d, device=self.device)
            )

            loss = expert_loss + agent_loss

            # disc_fn = partial(self.discriminator, policy=policy)
            if self._grad_pen != 0.0:
                n_expert = len(expert_batch["observations"])
                grad_pen = wass_grad_pen(
                    expert_batch["observations"],
                    expert_batch["actions"],
                    agent_batch["obs"][:n_expert],
                    agent_batch["action"][:n_expert],
                    False,
                    self.discriminator,
                )

                loss += self._grad_pen * grad_pen

            self.discrim_opt.zero_grad()
            loss.backward()
            self.discrim_opt.step()

            logger.collect_info("expert_loss", expert_loss.item())
            logger.collect_info("agent_loss", agent_loss.item())
            logger.collect_info("discim_loss", loss.item())

    def update(self, policy, rollouts, logger, **kwargs):
        if (
            self.reward_update_freq != -1
            and self._n_updates % self.reward_update_freq == 0
        ):
            self._update_discriminator(policy, rollouts, logger)

        obs, actions, next_obs, masks = extract_transition_batch(rollouts)
        with torch.no_grad():
            rollouts.rewards = self.discriminator.get_reward(
                cur_obs=obs,
                actions=actions,
                next_obs=next_obs,
                masks=masks,
                policy=policy,
            )
            self._ep_rewards = log_finished_rewards(rollouts, self._ep_rewards, logger)
        self.policy_updater.update(policy, rollouts, logger)
        self._n_updates += 1
