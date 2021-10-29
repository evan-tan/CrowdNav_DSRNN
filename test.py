import argparse
import logging
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
from importlib import import_module
from pathlib import Path

import torch
import torch.nn as nn

from crowd_sim import *
from crowd_sim.envs.utils.helper import unsqueeze
from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from pytorchBaselines.a2c_ppo_acktr.model import Policy
from pytorchBaselines.evaluation import evaluate


def main():
    # the following parameters will be determined for each test run
    test_parser = argparse.ArgumentParser("Parser for test.py", add_help=True)
    # the model directory that we are testing
    test_parser.add_argument("--model_dir", type=str, default="data/example_model")
    test_parser.add_argument("--viz", action="store_true")
    test_parser.add_argument(
        "--test_case",
        type=int,
        default=-1,
        help="if -1 is used, it will run 500 different cases; if specified, it will run the specified test case 5 times repeatedly",
    )
    # model weight file you want to test
    test_parser.add_argument("--test_model", type=str, default=None)
    test_parser.add_argument(
        "--test_name",
        type=str,
        default="test",
        help="name of experiment, to name .log file in test/",
    )
    test_parser.add_argument(
        "--default_config",
        type=bool,
        default=True,
        help="Whether or not to use default config used during training",
    )
    test_parser.add_argument(
        "--num_threads", type=int, default=1, help="Number of threads to allocate"
    )
    test_parser.add_argument(
        "--study_scenario",
        action="store_true",
        help="Whether or not to plot cum. rewards and distance to goal versus time when studying specific cases or scenarios",
    )
    test_args = test_parser.parse_args()

    model_dir_temp = test_args.model_dir
    if model_dir_temp.endswith("/"):
        model_dir_temp = model_dir_temp[:-1]

    # check which model to load by default
    if test_args.test_model is None:
        train_cfg_path = model_dir_temp.replace("/", ".") + ".configs.train_config"
        cfg_module = import_module(train_cfg_path)
        train_cfg = getattr(cfg_module, "Config")
        tmp_cfg = train_cfg()
        if tmp_cfg.training.num_processes == 12:
            # LOAD DEFAULT MODEL
            test_args.test_model = "27776.pt"
        else:
            # sort through all checkpoints
            checkpoints_path = Path(model_dir_temp) / "checkpoints"
            f_list = []
            for model in checkpoints_path.iterdir():
                # convert PosixPath to str
                checkpoints_path = str(checkpoints_path)
                model = str(model)
                model = model.replace(checkpoints_path, "").replace("/", "")
                f_list.append(model)
            test_args.test_model = sorted(f_list, reverse=True)[0]

    # import config class from saved directory
    # if not found, import from the default directory
    try:
        if test_args.default_config:
            cfg_file = "train_config"
            print("Using default train_config.py")
        model_dir_string = model_dir_temp.replace("/", ".") + ".configs." + cfg_file
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, "Config")
    except:
        print(
            f"Failed to get {test_args.default_config} from {test_args.model_dir}, loading default CrowdNav config"
        )
        from crowd_nav.configs.config import Config

    config = Config()

    # configure logging and device
    # print test result in log file
    log_dir = Path.cwd() / test_args.model_dir / "test"
    f_name = ""
    if not log_dir.exists():
        log_dir.mkdir()

    if test_args.test_model:
        f_name += "model_" + str(Path(test_args.test_model).with_suffix("")) + "#"
    if test_args.test_name:
        f_name += "test_" + test_args.test_name + "#"

    f_name += ".log"
    # replace arbitrary placeholder
    f_name = f_name.replace("#", "_")
    log_file = log_dir / f_name
    # convert PosixPath to str
    log_file = str(log_file)

    file_handler = logging.FileHandler(log_file, mode="w")
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if test_args.test_case != -1:
        config.env.test_size = 1
    test_cases_str = "all" if test_args.test_case == -1 else str(test_args.test_case)
    logging.info("Test Cases: " + test_cases_str)
    logging.info("robot FOV %f", config.robot.FOV * np.pi)
    logging.info("humans FOV %f", config.humans.FOV * np.pi)

    torch.manual_seed(config.env.seed)
    torch.cuda.manual_seed_all(config.env.seed)
    if config.training.cuda:
        if config.training.cuda_deterministic:
            # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    torch.set_num_threads(test_args.num_threads)
    device = torch.device("cuda" if config.training.cuda else "cpu")

    logging.info("Create other envs with new settings")

    if test_args.viz:
        fig, ax = plt.subplots(figsize=(9, 9))
        val = config.sim.square_width + 5
        ax.set_xlim(-val / 2, val / 2)
        ax.set_ylim(-val / 2, val / 2)
        ax.set_xlabel("x(m)", fontsize=16)
        ax.set_ylabel("y(m)", fontsize=16)
        plt.show(block=False)
        plt.pause(0.1)
    else:
        ax = None

    load_path = Path.cwd() / test_args.model_dir / "checkpoints" / test_args.test_model
    load_path = str(load_path)
    logging.info(f"Using model {load_path}")

    env_name = config.env.env_name
    recurrent_cell = "GRU"

    eval_dir = Path.cwd() / test_args.model_dir / "eval"
    if not eval_dir.exists():
        eval_dir.mkdir()

    render_fig = fig if test_args.viz else None
    envs = make_vec_envs(
        env_name,
        config.env.seed,
        1,
        config.reward.gamma,
        eval_dir,
        device,
        allow_early_resets=True,
        config=config,
        ax=ax,
        test_case=test_args.test_case,
        fig=render_fig,
    )

    actor_critic = Policy(
        envs.observation_space.spaces,  # pass the Dict into policy to parse
        envs.action_space,
        base_kwargs=config,
        base=config.robot.policy,
    )

    actor_critic.load_state_dict(torch.load(load_path, map_location=device))
    actor_critic.base.nenv = 1
    # allow the usage of multiple GPUs to increase the number of examples processed simultaneously
    nn.DataParallel(actor_critic).to(device)

    # actor_critic, ob_rms, eval_envs, num_processes, device, num_episodes
    raw_rewards, discounted_rewards, dist_to_goal = evaluate(
        actor_critic=actor_critic,
        ob_rms=False,
        eval_envs=envs,
        num_processes=1,
        device=device,
        config=config,  # defaults to 500, number of episodes to test
        logging=logging,
        visualize=test_args.viz,
        recurrent_type=recurrent_cell,
    )

    if test_args.study_scenario:
        # plot cumulative reward vs time step
        metrics = [raw_rewards, discounted_rewards, dist_to_goal]
        COLORS = ("b", "g", "r", "c", "m", "y", "k", "w")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel("Time Step")
        ax2.set_xlabel("Time Step")
        ax1.set_ylabel("Cumulative Rewards")
        ax2.set_ylabel("Distance To Goal")
        for i, metric in enumerate(metrics):
            for key in metric:
                # fetch from the correct bin
                if len(metric[key]) > 0:
                    time_ = np.linspace(0, len(metric[key][0]), len(metric[key][0]))
                    if i == 0:
                        y_label = "Cumulative Reward (Raw)"
                        f_name = "cr_raw"
                    elif i == 1:
                        y_label = "Cumulative Reward (Discounted)"
                        f_name = "cr_disc"
                    elif i == 2:
                        y_label = "Distance To Goal"
                        f_name = "d2g"
                    y_data = np.array(metric[key]) if i == 2 else np.cumsum(metric[key])
                    y_data = y_data.squeeze()
                    if i == 2:
                        ax2.set_title("Plot of Distance To Goal against Time")
                        ax2.plot(
                            time_, y_data, color=COLORS[i], linewidth=1, label=y_label
                        )
                    else:
                        ax1.set_title("Plot Cumulative Rewards against Time")
                        ax1.plot(
                            time_, y_data, color=COLORS[i], linewidth=1, label=y_label
                        )

        ax1.legend(loc="lower right", fontsize=12)
        ax2.legend(loc="lower left", fontsize=12)

        plot_dir = log_dir / "plots"
        if not plot_dir.exists():
            plot_dir.mkdir()

        fig1.savefig(
            f"{str(plot_dir)}/rewards_vs_time_{key}_case_{test_args.test_case}",
            dpi=1200,
        )
        fig2.savefig(
            f"{str(plot_dir)}/d2g_vs_time_{key}_case_{test_args.test_case}", dpi=1200
        )
        print(f"Saved plots to {str(plot_dir)}")


if __name__ == "__main__":
    main()
