import logging
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from crowd_nav.configs.config import Config
from crowd_sim import *
from crowd_sim.envs.utils.helper import create_events_dict, log_events_dict
from crowd_sim.envs.utils.info import Collision, Danger, Nothing, ReachGoal, Timeout
from pytorchBaselines.a2c_ppo_acktr import algo, utils
from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from pytorchBaselines.a2c_ppo_acktr.model import Policy
from pytorchBaselines.a2c_ppo_acktr.storage import RolloutStorage, SRNNRolloutStorage


def main():
    config = Config()

    # save policy to output_dir
    # additional check if I want to overwrite the directory
    output_dir = config.training.output_dir
    if Path(output_dir).exists() and config.training.overwrite:
        overwrite_prompt = input(f"{output_dir} exists, Overwrite directory?[y/n]")
        if "y" in overwrite_prompt:
            # delete an entire directory tree
            shutil.rmtree(output_dir)
        else:
            return
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()

    if output_dir.endswith("/"):
        output_dir = output_dir[:-1]

    # copy files over to trained model
    shutil.copytree("crowd_nav/configs", str(Path(output_dir) / "configs"))
    # save config.py as train_config.py
    shutil.move(
        str(Path(output_dir) / "configs" / "config.py"),
        str(Path(output_dir) / "configs" / "train_config.py"),
    )

    # create symlink that points to main config.py
    cwd = str(Path.cwd())
    if cwd[:-1] == "/":
        cwd = cwd[:-1]
    os.symlink(
        cwd + "/crowd_nav/configs/config.py",
        str(Path(output_dir) / "configs" / "config.py"),
    )

    # configure logging
    log_file = os.path.join(config.training.output_dir, "output.log")
    mode = "a" if config.training.resume else "w"
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = Config()

    torch.manual_seed(config.env.seed)
    torch.cuda.manual_seed_all(config.env.seed)
    if config.training.cuda and torch.cuda.is_available():
        if config.training.cuda_deterministic:
            # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    # torch.set_num_threads(config.training.num_threads)
    torch.set_num_threads(torch.get_num_threads())
    device = torch.device(
        "cuda" if config.training.cuda and torch.cuda.is_available() else "cpu"
    )

    logging.info("Create other envs with new settings")

    # For fastest training: use GRU
    env_name = config.env.env_name
    recurrent_cell = "GRU"

    if config.sim.render:
        fig, ax = plt.subplots(figsize=(7, 7))
        val = config.sim.square_width
        ax.set_xlim(-val / 2, val / 2)
        ax.set_ylim(-val / 2, val / 2)
        ax.set_xlabel("x(m)", fontsize=16)
        ax.set_ylabel("y(m)", fontsize=16)
        # plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
    else:
        ax = None

    if config.sim.render:
        config.training.num_processes = 1
        config.ppo.num_mini_batch = 1

    # create a manager env
    render_fig = fig if config.sim.render else None
    envs = make_vec_envs(
        env_name,
        config.env.seed,
        config.training.num_processes,
        config.reward.gamma,
        None,
        device,
        False,
        config=config,
        ax=ax,
        fig=render_fig,
    )
    if type(envs.observation_space) == gym.spaces.Dict:
        # for SRNN
        obs_space = envs.observation_space.spaces
    else:
        obs_space = envs.observation_space

    if config.robot.policy == "srnn":
        actor_critic = Policy(
            envs.observation_space.spaces,  # pass the Dict into policy to parse
            envs.action_space,
            base_kwargs=config,
            base=config.robot.policy,
        )

        rollouts = SRNNRolloutStorage(
            config.ppo.num_steps,
            config.training.num_processes,
            envs.observation_space.spaces,
            envs.action_space,
            config.SRNN.human_node_rnn_size,
            config.SRNN.human_human_edge_rnn_size,
            recurrent_cell_type=recurrent_cell,
        )
    elif config.robot.policy == "convgru":
        actor_critic = Policy(
            obs_space.shape,
            envs.action_space,
            base_kwargs=config,
            base=config.robot.policy,
        )

        rollouts = RolloutStorage(
            config.ppo.num_steps,
            config.training.num_processes,
            obs_space.shape,
            envs.action_space,
            actor_critic.recurrent_hidden_state_size,
        )

    if config.training.resume:  # retrieve the model if resume = True
        load_path = config.training.load_path
        actor_critic, _ = torch.load(load_path)

    # allow the usage of multiple GPUs to increase the number of examples processed simultaneously
    nn.DataParallel(actor_critic).to(device)

    agent = algo.PPO(
        actor_critic,
        config.ppo.clip_param,
        config.ppo.epoch,
        config.ppo.num_mini_batch,
        config.ppo.value_loss_coef,
        config.ppo.entropy_coef,
        lr=config.training.lr,
        eps=config.training.eps,
        max_grad_norm=config.training.max_grad_norm,
    )

    obs = envs.reset()
    if isinstance(obs, dict):
        for key in obs:
            rollouts.obs[key][0].copy_(obs[key])
    else:
        rollouts.obs[0].copy_(obs)

    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = (
        int(config.training.num_env_steps)
        // config.ppo.num_steps
        // config.training.num_processes
    )
    tboard_logdir = config.training.output_dir + "/events"
    tboard_logdir = tboard_logdir.replace("//", "/")
    writer = SummaryWriter(log_dir=tboard_logdir)
    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    num_events = create_events_dict(config)

    for j in range(num_updates):

        if config.training.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer,
                j,
                num_updates,
                config.training.lr,
            )

        for step in range(config.ppo.num_steps):
            # Sample actions
            with torch.no_grad():
                if config.robot.policy == "srnn":
                    rollouts_obs = {}
                    for key in rollouts.obs:
                        rollouts_obs[key] = rollouts.obs[key][step]
                    rollouts_hidden_s = {}
                    for key in rollouts.recurrent_hidden_states:
                        rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][
                            step
                        ]
                    (
                        value,
                        action,
                        action_log_prob,
                        recurrent_hidden_states,
                    ) = actor_critic.act(
                        rollouts_obs, rollouts_hidden_s, rollouts.masks[step]
                    )
                else:
                    (
                        value,
                        action,
                        action_log_prob,
                        recurrent_hidden_states,
                    ) = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )

            if config.sim.render:
                envs.render()
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # loop through infos since we are using vectorized environments
            for info in infos:
                curr_scenario = info.get("info").get("scenario")
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])
                if isinstance(info.get("info").get("event"), ReachGoal):
                    num_events["success"]["total"] += 1
                    num_events["success"][curr_scenario] += 1
                elif isinstance(info.get("info").get("event"), Collision):
                    num_events["collision"]["total"] += 1
                    num_events["collision"][curr_scenario] += 1
                elif isinstance(info.get("info").get("event"), Timeout):
                    num_events["timeout"]["total"] += 1
                    num_events["timeout"][curr_scenario] += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            if config.robot.policy == "srnn":
                rollouts_obs = {}
                for key in rollouts.obs:
                    rollouts_obs[key] = rollouts.obs[key][-1]
                rollouts_hidden_s = {}
                for key in rollouts.recurrent_hidden_states:
                    rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
                next_value = actor_critic.get_value(
                    rollouts_obs, rollouts_hidden_s, rollouts.masks[-1]
                ).detach()

            else:
                next_value = actor_critic.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()

        rollouts.compute_returns(
            next_value,
            config.ppo.use_gae,
            config.reward.gamma,
            config.ppo.gae_lambda,
            config.training.use_proper_time_limits,
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save the model for every interval-th episode or for the last epoch
        if j % config.training.save_interval == 0 or j == num_updates - 1:
            save_path = os.path.join(config.training.output_dir, "checkpoints")
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            # if you normalized the observation, you may also want to save rms
            # torch.save([
            # 	actor_critic,
            # 	getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            # ], os.path.join(save_path, '%.5i'%j + ".pt"))

            torch.save(
                actor_critic.state_dict(), os.path.join(save_path, "%.5i" % j + ".pt")
            )

        if j % config.training.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (
                (j + 1) * config.training.num_processes * config.ppo.num_steps
            )
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
                "{:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                )
            )

            df = pd.DataFrame(
                {
                    "misc/nupdates": [j],
                    "misc/total_timesteps": [total_num_steps],
                    "fps": int(total_num_steps / (end - start)),
                    "eprewmean": [np.mean(episode_rewards)],
                    "loss/policy_entropy": dist_entropy,
                    "loss/policy_loss": action_loss,
                    "loss/value_loss": value_loss,
                }
            )

            # tensorboard logs
            writer.add_scalar("mean_reward", np.mean(episode_rewards), total_num_steps)
            writer.add_scalar(
                "median_reward", np.median(episode_rewards), total_num_steps
            )
            writer.add_scalar("min_reward", np.min(episode_rewards), total_num_steps)
            writer.add_scalar("max_reward", np.max(episode_rewards), total_num_steps)
            writer.add_scalar(
                "policy_entropy (dist_entropy)", dist_entropy, total_num_steps
            )
            writer.add_scalar("policy_loss (action_loss)", action_loss, total_num_steps)
            writer.add_scalar("value_loss", value_loss, total_num_steps)

            if (
                os.path.exists(os.path.join(config.training.output_dir, "progress.csv"))
                and j > 20
            ):
                df.to_csv(
                    os.path.join(config.training.output_dir, "progress.csv"),
                    mode="a",
                    header=False,
                    index=False,
                )
            else:
                df.to_csv(
                    os.path.join(config.training.output_dir, "progress.csv"),
                    mode="w",
                    header=True,
                    index=False,
                )

    logging.info("")
    logging.info("SCENARIO BREAKDOWN: ")
    log_events_dict(num_events, logging)

    end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info("START @ " + start_time)
    logging.info("END @ " + end_time)


if __name__ == "__main__":
    main()
