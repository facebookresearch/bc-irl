# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable, List

import functorch
import higher
import torch
import torch.nn as nn
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from tensordict.tensordict import TensorDict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from imitation_learning.common.plotting import plot_actions
from imitation_learning.common.utils import log_finished_rewards


def reg_init(old_policy, **kwargs):
    return old_policy


class BCIRL(nn.Module):
    def __init__(
        self,
        reward: DictConfig,
        inner_updater: DictConfig,
        get_dataset_fn,
        batch_size: int,
        inner_opt: DictConfig,
        reward_opt: DictConfig,
        irl_loss: DictConfig,
        plot_interval: int,
        norm_expert_actions: bool,
        n_inner_iters: int,
        num_steps: int,
        reward_update_freq: int,
        device,
        total_num_updates: int,
        num_envs: int,
        use_lr_decay: bool,
        policy_init_fn: Callable[[nn.Module, nn.Module], nn.Module],
        force_num_env_steps_lr_decay: float = -1.0,
        **kwargs,
    ):
        super().__init__()
        self.inner_updater = instantiate(inner_updater)
        self.reward = instantiate(reward).to(device)

        self.dataset = call(get_dataset_fn)
        self.data_loader = DataLoader(self.dataset, batch_size, shuffle=True)

        self.inner_opt = inner_opt
        self.reward_opt = instantiate(reward_opt, params=self.reward.parameters())
        self._n_updates = 0
        self.use_lr_decay = use_lr_decay
        self.policy_init_fn = policy_init_fn

        if force_num_env_steps_lr_decay > 0:
            use_total_num_updates = force_num_env_steps_lr_decay // (
                num_envs * num_steps
            )
        else:
            use_total_num_updates = total_num_updates

        self.lr_scheduler = LambdaLR(
            optimizer=self.reward_opt,
            lr_lambda=lambda x: 1 - (self._n_updates / use_total_num_updates),
        )

        self.irl_loss = instantiate(irl_loss)
        self.data_loader_iter = iter(self.data_loader)

        self.plot_interval = plot_interval
        self.norm_expert_actions = norm_expert_actions
        self.n_inner_iters = n_inner_iters
        self.num_steps = num_steps
        self.reward_update_freq = reward_update_freq
        self.device = device
        self.num_envs = num_envs
        self._ep_rewards = torch.zeros(num_envs, device=self.device)

    def state_dict(self):
        return {
            **super().state_dict(),
            "reward_opt": self.reward_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, should_load_opt):
        opt_state = state_dict.pop("reward_opt")
        if should_load_opt:
            self.reward_opt.load_state_dict(opt_state)
        return super().load_state_dict(state_dict)

    def viz_reward(self, cur_obs=None, action=None, next_obs=None) -> torch.Tensor:
        return self.reward(cur_obs, action, next_obs)

    def _irl_loss_step(self, policy, logger):
        # Compute the outer loop loss
        expert_batch = next(self.data_loader_iter, None)
        if expert_batch is None:
            self.data_loader_iter = iter(self.data_loader)
            expert_batch = next(self.data_loader_iter, None)
        expert_actions = expert_batch["actions"].to(self.device)
        expert_obs = expert_batch["observations"].to(self.device)
        if self.norm_expert_actions:
            # Clip expert actions to be within [-1,1]. Actions have no effect
            # out of that range
            expert_actions = torch.clamp(expert_actions, -1.0, 1.0)

        td = TensorDict(source={"observation": expert_obs}, batch_size=[])
        dist = policy.get_action_dist(td)
        # Ignore the variance prediction.
        pred_actions = dist.mean

        irl_loss_val = self.irl_loss(expert_actions, pred_actions)
        irl_loss_val.backward(retain_graph=True)

        logger.collect_info("irl_loss", irl_loss_val.item())

        if self._n_updates % self.plot_interval == 0:
            plot_actions(
                pred_actions.detach().cpu(),
                expert_actions.detach().cpu(),
                self._n_updates,
                logger.vid_dir,
            )

    @property
    def inner_lr(self):
        return self.inner_opt["lr"]

    def update(self, policy, rollouts, logger, envs):
        self.reward_opt.zero_grad()

        policy = call(self.policy_init_fn, old_policy=policy).to(self.device)
        policy_opt = instantiate(
            self.inner_opt, lr=self.inner_lr, params=policy.parameters()
        )

        # Setup meta learning loop
        with higher.innerloop_ctx(
            policy,
            policy_opt,
        ) as (dpolicy, diffopt):
            for inner_i in range(self.n_inner_iters):

                rollouts["reward"] = self.reward(
                    rollouts["observation"],
                    rollouts["action"],
                    rollouts["next_observation"],
                )

                if inner_i == 0:
                    self._ep_rewards = log_finished_rewards(
                        rollouts, self._ep_rewards, logger
                    )

                # Inner loop policy update
                self.inner_updater.update(dpolicy, rollouts, logger, diffopt)

                if inner_i != self.n_inner_iters - 1:
                    # Additional inner loop policy updates (if n_inner_iters > 1)
                    td = rollouts[:, -1]
                    new_rollouts = TensorDict(
                        {},
                        batch_size=[self.num_envs, self.num_steps],
                        device=self.device,
                    )
                    for step_idx in range(self.num_steps):
                        with torch.no_grad():
                            policy.act(td)
                        next_obs, reward, done, infos = envs.step(td["action"])

                        td["next_observation"] = next_obs
                        for env_i, info in enumerate(infos):
                            if "final_obs" in info:
                                td["next_observation"][env_i] = info["final_obs"]
                        td["reward"] = reward
                        td["done"] = done

                        new_rollouts[:, step_idx] = td
                        td["observation"] = next_obs
                        logger.collect_env_step_info(infos)
                    rollouts = new_rollouts

            # Compute IRL loss
            self._irl_loss_step(dpolicy, logger)

        if (
            self.reward_update_freq != -1
            and self._n_updates % self.reward_update_freq == 0
        ):
            self.reward_opt.step()
            if hasattr(self.reward, "log"):
                self.reward.log(logger)

        # Copy over the temporary policy form the meta-learning loop to the permanent policy.
        policy.load_state_dict(dpolicy.state_dict())

        if self.use_lr_decay and self.reward_update_freq != -1:
            # Step even if we did not update so we properly decay to 0.
            self.lr_scheduler.step()
            logger.collect_info("reward_lr", self.lr_scheduler.get_last_lr()[0])

        self._n_updates += 1
