# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from higher.optim import DifferentiableOptimizer
from hydra.utils import instantiate
from omegaconf import DictConfig


class DifferentiablePPO(nn.Module):
    def __init__(
        self,
        use_gae: bool,
        gae_lambda: float,
        gamma: float,
        use_clipped_value_loss: bool,
        clip_param: bool,
        value_loss_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        num_mini_batch: int,
        num_epochs: int,
    ):
        super().__init__()
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_mini_batch = num_mini_batch
        self.num_epochs = num_epochs
        self.value_loss_coef = value_loss_coef

    def update(
        self,
        policy,
        storage,
        logger,
        optimizer: DifferentiableOptimizer,
        rewards,
    ) -> None:
        with torch.no_grad():
            last_value = policy.get_value(
                storage.get_obs(-1),
                storage.recurrent_hidden_states[-1],
                storage.masks[-1],
            )

        advantages, returns = self.compute_derived(
            rewards,
            storage.masks,
            storage.bad_masks,
            storage.value_preds.detach(),
            last_value,
        )

        for _ in range(self.num_epochs):
            data_gen = storage.data_generator(
                self.num_mini_batch, returns=returns[:-1], advantages=advantages
            )
            for sample in data_gen:
                ac_eval = policy.evaluate_actions(
                    sample["obs"],
                    sample["hxs"],
                    sample["mask"],
                    sample["action"],
                )

                ratio = torch.exp(ac_eval["log_prob"] - sample["prev_log_prob"])
                surr1 = ratio * sample["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                    * sample["advantages"]
                )
                action_loss = -torch.min(surr1, surr2).mean(0)

                value_target = sample["returns"].detach()

                if self.use_clipped_value_loss:
                    value_pred_clipped = sample["value"] + (
                        ac_eval["value"] - sample["value"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (ac_eval["value"] - value_target).pow(2)
                    value_losses_clipped = (value_pred_clipped - value_target).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (value_target - ac_eval["value"]).pow(2).mean()

                loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - ac_eval["dist_entropy"].mean() * self.entropy_coef
                )

                # if self.max_grad_norm > 0:
                #     nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                optimizer.step(loss)

                logger.collect_info("value_loss", value_loss.mean().item())
                logger.collect_info("action_loss", action_loss.mean().item())
                logger.collect_info(
                    "dist_entropy", ac_eval["dist_entropy"].mean().item()
                )

    def compute_derived(
        self,
        rewards,
        masks,
        bad_masks,
        value_preds,
        last_value,
    ):
        num_steps, num_envs = rewards.shape[:2]
        returns = torch.zeros(num_steps + 1, num_envs, 1, device=last_value.device)
        if self.use_gae:
            value_preds[-1] = last_value
            gae = 0
            for step in reversed(range(rewards.size(0))):
                delta = (
                    rewards[step]
                    + self.gamma * value_preds[step + 1] * masks[step + 1]
                    - value_preds[step]
                )
                gae = delta + self.gamma * self.gae_lambda * masks[step + 1] * gae
                gae = gae * bad_masks[step + 1]
                returns[step] = gae + value_preds[step]
        else:
            returns[-1] = last_value
            for step in reversed(range(rewards.size(0))):
                returns[step] = (
                    returns[step + 1] * self.gamma * masks[step + 1] + rewards[step]
                ) * bad_masks[step + 1] + (1 - bad_masks[step + 1]) * value_preds[step]
        advantages = returns[:-1] - value_preds[:-1]

        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        return advantages, returns
