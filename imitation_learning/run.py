# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import random
from typing import Dict

import gym.spaces as spaces
import hydra
import numpy as np
import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
from rl_utils.envs import create_vectorized_envs
from rl_utils.logging import Logger
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import step_mdp

from imitation_learning.common.evaluator import Evaluator


def set_seed(seed: int) -> None:
    """
    Sets the seed for numpy, python random, and pytorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="config", config_name="default")
def main(cfg) -> Dict[str, float]:
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    # Setup the environments
    set_env_settings = {
        k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
        for k, v in cfg.env.env_settings.items()
    }
    envs = create_vectorized_envs(
        cfg.env.env_name,
        cfg.num_envs,
        seed=cfg.seed,
        device=device,
        **set_env_settings,
    )

    steps_per_update = cfg.num_steps * cfg.num_envs
    num_updates = int(cfg.num_env_steps) // steps_per_update

    # Set dynamic variables in the config.
    cfg.obs_shape = envs.observation_space.shape
    cfg.action_dim = envs.action_space.shape[0]
    cfg.action_is_discrete = isinstance(cfg.action_dim, spaces.Discrete)
    cfg.total_num_updates = num_updates

    logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)
    policy = hydra_instantiate(cfg.policy)
    policy = policy.to(device)
    updater = hydra_instantiate(cfg.policy_updater, policy=policy, device=device)
    evaluator: Evaluator = hydra_instantiate(
        cfg.evaluator,
        envs=envs,
        vid_dir=logger.vid_path,
        updater=updater,
        logger=logger,
        device=device,
    )

    start_update = 0
    if cfg.load_checkpoint is not None:
        # Load a checkpoint for the policy/reward. Also potentially resume
        # training.
        ckpt = torch.load(cfg.load_checkpoint)
        updater.load_state_dict(ckpt["updater"], should_load_opt=cfg.resume_training)
        if cfg.load_policy:
            policy.load_state_dict(ckpt["policy"])
        if cfg.resume_training:
            start_update = ckpt["update_i"] + 1

    eval_info = {"run_name": logger.run_name}

    if cfg.only_eval:
        # Evaluate the policy and end.
        eval_result = evaluator.evaluate(policy, cfg.num_eval_episodes, 0)
        logger.collect_infos(eval_result, "eval.", no_rolling_window=True)
        eval_info.update(eval_result)
        logger.interval_log(0, 0)
        logger.close()

        return eval_info

    obs = envs.reset()
    td = TensorDict({"observation": obs}, batch_size=[cfg.num_envs])
    # Storage for the rollouts
    storage_td = TensorDict({}, batch_size=[cfg.num_envs, cfg.num_steps], device=device)

    for update_i in range(start_update, num_updates):
        is_last_update = update_i == num_updates - 1
        for step_idx in range(cfg.num_steps):
            # Collect experience.
            with torch.no_grad():
                policy.act(td)
            next_obs, reward, done, infos = envs.step(td["action"])

            td["next_observation"] = next_obs
            for env_i, info in enumerate(infos):
                if "final_obs" in info:
                    td["next_observation"][env_i] = info["final_obs"]
            td["reward"] = reward
            td["done"] = done

            storage_td[:, step_idx] = td
            td["observation"] = next_obs
            # Log to CLI/wandb.
            logger.collect_env_step_info(infos)

        # Call method specific update function
        updater.update(policy, storage_td, logger, envs=envs)

        if cfg.eval_interval != -1 and (
            update_i % cfg.eval_interval == 0 or is_last_update
        ):
            with torch.no_grad():
                eval_result = evaluator.evaluate(
                    policy, cfg.num_eval_episodes, update_i
                )
            logger.collect_infos(eval_result, "eval.", no_rolling_window=True)
            eval_info.update(eval_result)

        if cfg.log_interval != -1 and (
            update_i % cfg.log_interval == 0 or is_last_update
        ):
            logger.interval_log(update_i, steps_per_update * (update_i + 1))

        if cfg.save_interval != -1 and (
            (update_i + 1) % cfg.save_interval == 0 or is_last_update
        ):
            save_name = osp.join(logger.save_path, f"ckpt.{update_i}.pth")
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "updater": updater.state_dict(),
                    "update_i": update_i,
                },
                save_name,
            )
            print(f"Saved to {save_name}")
            eval_info["last_ckpt"] = save_name

    logger.close()
    return eval_info


if __name__ == "__main__":
    main()
