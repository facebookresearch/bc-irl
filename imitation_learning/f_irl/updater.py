# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from rl_utils.common.net_utils import make_mlp_layers
from torch.utils.data import DataLoader

from imitation_learning.common.plotting import plot_actions
from imitation_learning.common.utils import (create_next_obs,
                                             extract_transition_batch,
                                             log_finished_rewards)
from imitation_learning.gail.updater import GAIL


class fIRL(GAIL):
    """
    From https://github.com/twni2016/f-IRL/blob/a3f1ec66f29c6d659abba630f70f8ae2e59ebe1e/firl/divs/f_div_disc.py
    """

    def __init__(
        self,
        discriminator: DictConfig,
        reward: DictConfig,
        policy_updater: DictConfig,
        get_dataset_fn,
        batch_size: int,
        num_discrim_batches: int,
        reward_opt: DictConfig,
        discrim_opt: DictConfig,
        reward_update_freq: int,
        importance_sampling: bool,
        div_type: str,
        device,
        policy,
        num_envs,
        **kwargs,
    ):
        super().__init__(
            discriminator,
            policy_updater,
            get_dataset_fn,
            batch_size,
            num_discrim_batches,
            discrim_opt,
            reward_update_freq,
            device,
            policy,
            num_envs,
        )
        self.reward = instantiate(reward).to(device)
        self._div_type = div_type
        self._importance_sampling = importance_sampling
        self.reward_opt = instantiate(reward_opt, params=self.reward.parameters())

    def state_dict(self, **kwargs):
        return {
            **super().state_dict(**kwargs),
            "reward_opt": self.reward_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, should_load_opt):
        opt_state = state_dict.pop("reward_opt")
        if should_load_opt:
            self.reward_opt.load_state_dict(opt_state)
        return super().load_state_dict(state_dict, should_load_opt)

    def viz_reward(self, cur_obs=None, action=None, next_obs=None) -> torch.Tensor:
        # Intentional to assign cur obs to next obs so we show reward for that state.
        return self.reward(next_obs=cur_obs)

    def _update_reward(self, policy, rollouts, logger):
        agent_data = self._get_agent_samples(rollouts)
        for expert_batch, agent_batch in zip(self.expert_data, agent_data):
            # Combine experience from both.
            with torch.no_grad():
                obs = torch.cat(
                    [
                        expert_batch["next_observations"],
                        agent_batch["next_obs"],
                    ],
                    0,
                )
                actions = torch.cat([expert_batch["actions"], agent_batch["action"]], 0)
                logits = self.discriminator(cur_obs=obs)

            # JS
            if self._div_type == "fkl":
                t1 = torch.exp(logits)
            elif self._div_type == "rkl":
                t1 = logits
            elif self._div_type == "js":
                t1 = F.softplus(logits)
            else:
                raise ValueError()
            t1 = -t1
            t2 = self.reward(next_obs=obs)

            if self._importance_sampling:
                with torch.no_grad():
                    traj_reward = t2.detach().clone()
                    traj_log_prob = policy.evaluate_actions(obs, None, None, actions)[
                        "log_prob"
                    ]
                    IS_ratio = F.softmax(traj_reward - traj_log_prob, dim=0)
                loss = (IS_ratio * t1 * t2).mean() - (
                    (IS_ratio * t1).mean() * (IS_ratio * t2).mean()
                )
            else:
                loss = (t1 * t2).mean() - (t1.mean() * t2.mean())

            self.reward_opt.zero_grad()
            loss.backward()
            self.reward_opt.step()
            logger.collect_info("reward_loss", loss.item())

    def update(self, policy, rollouts, logger, **kwargs):
        if (
            self.reward_update_freq != -1
            and self._n_updates % self.reward_update_freq == 0
        ):
            self._update_discriminator(policy, rollouts, logger)
            self._update_reward(policy, rollouts, logger)

        obs, actions, next_obs, masks = extract_transition_batch(rollouts)
        with torch.no_grad():
            rollouts.rewards = self.reward(next_obs=next_obs)
            self._ep_rewards = log_finished_rewards(rollouts, self._ep_rewards, logger)
        self.policy_updater.update(policy, rollouts, logger)
        self._n_updates += 1
