import time
from copy import deepcopy

import numpy as np
import torch
from crowd_sim.envs.utils.info import Collision, Danger, Nothing, ReachGoal, Timeout
from matplotlib import pyplot as plt

from pytorchBaselines.a2c_ppo_acktr import utils
from pytorchBaselines.metrics import Metrics


def evaluate(
    actor_critic,
    ob_rms,
    eval_envs,
    num_processes,
    device,
    config,
    logging,
    visualize=False,
    recurrent_type="GRU",
):
    if ob_rms:
        vec_norm = utils.get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

    rnn_factor = 2 if recurrent_type == "LSTM" else 1
    eval_recurrent_hidden_states = {}
    node_num = 1
    edge_num = actor_critic.base.human_num + 1

    eval_recurrent_hidden_states["human_node_rnn"] = torch.zeros(
        num_processes,
        node_num,
        config.SRNN.human_node_rnn_size * rnn_factor,
        device=device,
    )

    eval_recurrent_hidden_states["human_human_edge_rnn"] = torch.zeros(
        num_processes,
        edge_num,
        config.SRNN.human_human_edge_rnn_size * rnn_factor,
        device=device,
    )

    eval_masks = torch.zeros(num_processes, 1, device=device)
    metrics = Metrics(logging)
    # metrics across all episodes
    success_times = []
    success_cases = []

    collision_times = []
    collision_cases = []

    timeout_times = []
    timeout_cases = []

    metric_bin = {"success": [], "collision": [], "timeout": []}
    raw_rewards = deepcopy(metric_bin)
    discounted_rewards = deepcopy(metric_bin)
    dist_to_goal = deepcopy(metric_bin)

    path_lengths = []
    chc_total = []  # cumulative heading change
    min_dist = []
    gamma = 0.99
    base_env = eval_envs.venv.envs[0].env

    # social metrics
    personal_violation_times = []
    path_violation_times = []
    aggregate_nav_times = []
    jerk_costs = []
    speed_violation_times = []

    # START ALL SIDE PREFERENCE EPISODES
    if config.test.side_preference:
        side_preferences = {
            "passing": {"left": 0, "right": 0},
            "overtaking": {"left": 0, "right": 0},
            "crossing": {"left": 0, "right": 0},
        }
        side_counter = {"left": 0, "right": 0}
        scenario = config.test.side_preference_scenario

    obs = eval_envs.reset()
    test_size = config.env.test_size if not config.test.side_preference else 200
    for k in range(test_size):
        done = False
        episode_rewards = []
        episode_d2g = []  # episode distance to goal
        step_counter = 0

        global_time = 0.0
        episode_path = 0.0
        episode_chc = 0.0  # cumulative heading change

        personal_violation_time = 0
        path_violation_time = 0
        jerk_cost = 0
        speed_violation_time = 0
        aggregate_nav_time = 0

        last_pos = obs["robot_node"][0, 0, 0:2].cpu().numpy()  # robot px, py
        last_angle = np.arctan2(
            obs["temporal_edges"][0, 0, 1].cpu().numpy(),
            obs["temporal_edges"][0, 0, 0].cpu().numpy(),
        )  # robot theta

        total_render_time = 0
        while not done:
            step_counter += 1
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True,
                )
            if not done:
                global_time = base_env.global_time
            if visualize:
                start = time.time()
                eval_envs.render()
                total_render_time += time.time() - start

            # Obser reward and next obs
            obs, step_reward, done, step_info = eval_envs.step(action)

            episode_path += np.linalg.norm(
                np.array(
                    [
                        last_pos[0] - obs["robot_node"][0, 0, 0].cpu().numpy(),
                        last_pos[1] - obs["robot_node"][0, 0, 1].cpu().numpy(),
                    ]
                )
            )

            cur_angle = np.arctan2(
                obs["temporal_edges"][0, 0, 1].cpu().numpy(),
                obs["temporal_edges"][0, 0, 0].cpu().numpy(),
            )

            episode_chc += abs(cur_angle - last_angle)

            last_pos = obs["robot_node"][0, 0, 0:2].cpu().numpy()  # robot px, py
            last_angle = cur_angle

            if isinstance(step_info[0]["info"], Danger):
                min_dist.append(step_info[0]["info"].min_dist)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device,
            )
            episode_rewards.append(step_reward.item())

            if config.test.side_preference:
                if step_info[0].get("info").get(scenario) is not None:
                    curr_scenario = step_info[0].get("info").get(scenario)
                    if curr_scenario.get("left") == 1:
                        side_counter["left"] += 1
                    elif curr_scenario.get("right") == 1:
                        side_counter["right"] += 1

            if step_info[0].get("info").get("personal_violation") == 1:
                personal_violation_time += base_env.time_step
            if step_info[0].get("info").get("path_violation"):
                path_violation_time += base_env.time_step * step_info[0].get(
                    "info"
                ).get("path_violation")
            if step_info[0].get("info").get("aggregate_nav_time"):
                aggregate_nav_time += base_env.time_step * step_info[0].get("info").get(
                    "aggregate_nav_time"
                )
            if step_info[0].get("info").get("jerk_cost"):
                jerk_cost += step_info[0]["info"].get("jerk_cost")
            if step_info[0].get("info").get("speed_violation") == 1:
                speed_violation_time += base_env.time_step

            if step_info[0].get("info").get("dist_to_goal"):
                episode_d2g.append(step_info[0].get("info").get("dist_to_goal"))

        # END OF SINGLE EPISODE

        print("")
        print("Episode", k, "ends in", step_counter, "steps")
        if visualize:
            print(f"Average FPS = {1 / (total_render_time / step_counter):.3f}")

        # multiple steps in each episode, but we want to label with single value. calculate number of occurrences of left and right and take highest
        if config.test.side_preference:
            for k in side_counter.keys():
                side_counter[k] /= step_counter

        tmp_disc_reward = [
            pow(gamma, t * base_env.robot.time_step * base_env.robot.v_pref) * reward
            for t, reward in enumerate(episode_rewards)
        ]

        if isinstance(step_info[0].get("info").get("event"), ReachGoal):
            success_times.append(global_time)
            success_cases.append(k)

            discounted_rewards["success"].append(tmp_disc_reward)
            raw_rewards["success"].append(episode_rewards)
            dist_to_goal["success"].append(episode_d2g)
            chc_total.append(episode_chc)
            path_lengths.append(episode_path)

            personal_violation_times.append(personal_violation_time)
            path_violation_times.append(path_violation_time)
            aggregate_nav_times.append(aggregate_nav_time)
            jerk_costs.append(jerk_cost)
            speed_violation_times.append(speed_violation_time)

            if config.test.side_preference:
                if side_counter["left"] > side_counter["right"]:
                    side_preferences[scenario]["left"] += 1
                elif side_counter["left"] < side_counter["right"]:
                    side_preferences[scenario]["right"] += 1

            print("Success")
        elif isinstance(step_info[0].get("info").get("event"), Collision):
            collision_cases.append(k)
            collision_times.append(global_time)
            discounted_rewards["collision"].append(tmp_disc_reward)
            raw_rewards["collision"].append(episode_rewards)
            dist_to_goal["collision"].append(episode_d2g)
            print("Collision")
        elif isinstance(step_info[0].get("info").get("event"), Timeout):
            timeout_cases.append(k)
            timeout_times.append(base_env.time_limit)
            discounted_rewards["timeout"].append(tmp_disc_reward)
            raw_rewards["timeout"].append(episode_rewards)
            dist_to_goal["timeout"].append(episode_d2g)
            print("Time out")
        else:
            raise ValueError("Invalid end signal from environment")

        # reward tensor
        print(f"Reward={sum(episode_rewards)}")
        print(f"Path Length: {episode_path:.2f}")
        print(f"Time Taken: {global_time:.2f}")

    # END OF ALL EPISODES

    success_rate = len(success_times) / test_size
    collision_rate = len(collision_times) / test_size
    timeout_rate = len(timeout_times) / test_size
    assert len(success_times) + len(collision_times) + len(timeout_times) == test_size

    phase = "test"
    logging.info(f"{phase.upper()}")

    if phase in ["val", "test"]:
        total_time = sum(success_times + collision_times + timeout_times)
        if len(min_dist) > 0:
            avg_min_dist = np.mean(min_dist)
        else:
            avg_min_dist = float("nan")
        logging.info(
            f"Total time in danger: {(len(min_dist) * base_env.robot.time_step / total_time):.4f}, average min distance in danger: {avg_min_dist:.4f}"
        )

    logging.info(f"success rate: {success_rate:.3f}")
    logging.info(f"collision rate: {collision_rate:.3f}")
    logging.info(f"timeout rate: {timeout_rate:.3f}")

    logging.info("Success cases: " + " ".join([str(x) for x in success_cases]))
    logging.info("Collision cases: " + " ".join([str(x) for x in collision_cases]))
    logging.info("Timeout cases: " + " ".join([str(x) for x in timeout_cases]))

    metrics.add_metric("navigation time", success_times)
    metrics.add_metric("path length", path_lengths)

    mean_discounted_rew = []
    mean_raw_rew = []
    for key in raw_rewards:
        # sum step rewards across all episodes
        for ep_raw_rewards in raw_rewards[key]:
            mean_raw_rew.append(sum(ep_raw_rewards))
        for ep_disc_rewards in discounted_rewards[key]:
            mean_discounted_rew.append(sum(ep_disc_rewards))

    metrics.add_metric("discounted reward", mean_discounted_rew)
    metrics.add_metric("non-discounted rewards", mean_raw_rew)
    metrics.add_metric("cumulative heading change", chc_total)

    # social conformity metrics
    metrics.add_metric("SM1 - personal space violation", personal_violation_times)
    metrics.add_metric("SM2 - path violation", path_violation_times)
    metrics.add_metric("SM3 - aggregate time", aggregate_nav_times)
    metrics.add_metric("SM4 - jerk cost", jerk_costs)
    metrics.add_metric("SM5 - speed violation", speed_violation_times)
    if config.test.side_preference:
        left_percentage = side_preferences[scenario]["left"] / test_size
        right_percentage = side_preferences[scenario]["right"] / test_size
        logging.info(f"Side Preference - {scenario}")
        logging.info(f"Left % = {100*left_percentage:.3f}%")
        logging.info(f"Right % = {100*right_percentage:.3f}%")

    metrics.log_metrics()

    eval_envs.close()

    return raw_rewards, discounted_rewards, dist_to_goal
