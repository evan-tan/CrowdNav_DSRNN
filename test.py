import logging
import argparse
import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import torch
import torch.nn as nn
from pathlib import Path
from importlib import import_module
from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from pytorchBaselines.evaluation import evaluate
from crowd_sim import *
from pytorchBaselines.a2c_ppo_acktr.model import Policy


def main():
    # the following parameters will be determined for each test run
    test_parser = argparse.ArgumentParser("Parser for test.py", add_help=True)
    # the model directory that we are testing
    test_parser.add_argument("--model_dir", type=str, default="data/example_model")
    test_parser.add_argument("--visualize", default=True, action="store_true")
    test_parser.add_argument(
        "--test_case",
        type=int,
        default=-1,
        help="if -1 is used, it will run 500 different cases; if >=0, it will run the specified test case repeatedly",
    )
    # model weight file you want to test
    test_parser.add_argument("--test_model", type=str, default="27776.pt")
    test_parser.add_argument(
        "--test_name",
        type=str,
        default="test",
        help="name of experiment, to name .log file in test/",
    )
    test_parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Name of config module to load. Use config to use config.py, or leave blank to use train_config",
    )
    test_args = test_parser.parse_args()

    model_dir_temp = test_args.model_dir
    if model_dir_temp.endswith("/"):
        model_dir_temp = model_dir_temp[:-1]
    # import config class from saved directory
    # if not found, import from the default directory
    try:
        if len(test_args.config) > 0:
            cfg_file = test_args.config
            print("Using config.py symbolic link")
        else:
            cfg_file = "train_config"
            print("Using default train_config.py")

        model_dir_string = model_dir_temp.replace("/", ".") + ".configs." + cfg_file
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, "Config")
    except:
        print(
            f"Failed to get {test_args.config} from {test_args.model_dir}, loading default CrowdNav config"
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

    if test_args.visualize:
        f_name += "visual"

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

    torch.set_num_threads(1)
    device = torch.device("cuda" if config.training.cuda else "cpu")

    logging.info("Create other envs with new settings")

    if test_args.visualize:
        fig, ax = plt.subplots(figsize=(7, 7))
        val = config.sim.square_width
        ax.set_xlim(-val, val)
        ax.set_ylim(-val, val)
        ax.set_xlabel("x(m)", fontsize=16)
        ax.set_ylabel("y(m)", fontsize=16)
        plt.ion()
        plt.show()
    else:
        ax = None

    load_path = Path.cwd() / test_args.model_dir / "checkpoints" / test_args.test_model
    load_path = str(load_path)
    print(f"Using model {load_path}")

    env_name = config.env.env_name
    recurrent_cell = "GRU"

    eval_dir = Path.cwd() / test_args.model_dir / "eval"
    if not eval_dir.exists():
        eval_dir.mkdir()

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
    )

    actor_critic = Policy(
        envs.observation_space.spaces,  # pass the Dict into policy to parse
        envs.action_space,
        base_kwargs=config,
        base=config.robot.policy)

    actor_critic.load_state_dict(torch.load(load_path, map_location=device))
    actor_critic.base.nenv = 1
    # allow the usage of multiple GPUs to increase the number of examples processed simultaneously
    nn.DataParallel(actor_critic).to(device)

    # actor_critic, ob_rms, eval_envs, num_processes, device, num_episodes
    evaluate(
        actor_critic=actor_critic,
        ob_rms=False,
        eval_envs=envs,
        num_processes=1,
        device=device,
        config=config,  # defaults to 500, number of episodes to test
        logging=logging,
        visualize=test_args.visualize,
        recurrent_type=recurrent_cell,
    )


if __name__ == '__main__':
    main()
