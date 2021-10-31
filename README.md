# CrowdNav-DSRNN
the original README is renamed as og_README.md

Use [find_checkpoints.py](find_checkpoints.py) to easily find checkpoints after training a model, rather than inspecting by hand. Before training, please read the options in [config.py](crowd_nav/configs/config.py) carefully!

# Setup
Inside `setup/` run `./full_setup.sh`, make sure to change the variables... **REPOS_PATH, ENV_NAME, and PYTHON_VER** before running the script

# [config.py](crowd_nav/configs/config.py)

## What does this file do?

All of the options are documented in [config.py](crowd_nav/configs/config.py). These detail the following options that
are available during testing/training etc. During training, a snapshot of the [config.py](crowd_nav/configs/config.py)
is taken and saved as **train_config.py** inside `$(pwd)/data/model_name/configs/train_config.py`

## Why are there calculations inside config?

I've tried to put as much stuff inside config.py instead of code so that it's really **what you see is what you get**
rather than having to search for some variable in some obscure file.

# Scenario Selection

Scenario lists are built using `sim.train_val_sim` during training and `sim.test_sim` during testing. Each scenario is equally weighted and randomly chosen during `CrowdSimDict.reset()`.

To test **side preference**, change `sim.test_sim` to a **SINGLE ELEMENT LIST** containing your scenario. To test **social metrics**, set `test.social_metrics = True` in [config.py](crowd_nav/configs/config.py)

If testing **social metrics** the order of testing each scenario is sequential, i.e. circle crossing, random crossing,
parallel traffic, perpendicular traffic, circle crossing, etc. This is done in `crowd_sim_dict.py`.

# Testing

[Kevin's paper: Social Conformity Metrics](https://hri-methods-metrics.github.io/abstract_2021/Social_Conformity_Evaluation_of_Deep_Reinforcement_Learning_Crowd_Navigation_Algorithms%20-%20Junxian%20Wang.pdf)
*NOTE: NOT UPDATED WITH SUBMITTED PAPER YET AS OF 30 OCT 2021*

- To evaluate a model normally for **social metrics 1-5**,
    - set `test.social_metrics = True`
        - this boosts number of test episodes 500 -> 2000
        - changes sim.circle_radius 6 -> 4
        - Robot behaviour: fix spawn and goal every episode instead of being random (
          see `CrowdSim.generate_robot_humans()`)
        - Agent behaviour: depends on `sim.test_sim`

- To test **social metric 6** (side preference)
    - set `sim.test_sim = [x]`, where x is in `[side_pref_passing, side_pref_crossing, side_pref_overtaking]`
        - this changes number of test episodes 500 -> 200
        - this spawns 1 agent, and 1 robot within the simulator by calling `CrowdSim.create_agent_attributes()`
        - Robot behaviour: fix spawn and goal every episode instead of being random (
          see `CrowdSim.generate_robot_humans()`)
        - Agent behaviour: depends on `sim.test_sim`
- To view distance to goal VS time and the corresponding cumulative discounted/non-discounted rewards, use the `--study_scenario` flag in [test.py](test.py)

# Training

Currently there are 4 available scenarios to enable during training. See [Scenario Selection](#scenario-selection)
the training output will be available in `data/model_name/output.log`

# Phase

`self.phase = "train" or "test"` is inferred during creation of the vectorized environment. See `env.phase = "train"` in
Line 70-73 of [envs.py](pytorchBaselines/a2c_ppo_acktr/envs.py)

# Log Files

## Testing

A logfile will be created in data/model_name/test where the file name has the following
format: `model_[checkpoint number]_test_[test name]_.log`, see the argument parser in `test.py` for full details

## Training

The training output will be available in `data/model_name/output.log`

# How robots and humans are spawned
The Function calls are roughly
`CrowdSimDict.reset() -> CrowdSim.generate_robot_humans() -> CrowdSim.generate_robot_humans() -> CrowdSim.generate_circle_crossing_human() -> CrowdSim.create_agent_attributes()`
...


`reset()` is called in `crowd_sim_dict()`, which is a child class of the CrowdSim class. `reset()` calls CrowdSim's
method `generate_robot_humans()` to spawn the robot **first**, then `generate_random_human_position()` is called where
approximately n calls are made to `generate_circle_crossing_human()`. all human and agent positions and goals.
