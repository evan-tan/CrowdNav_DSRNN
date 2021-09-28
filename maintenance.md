# How robots and humans are spawned
## Function calls
TLDR: CrowdSimDict.reset() -> CrowdSim.generate_robot_humans() -> CrowdSim.generate_robot_humans() -> CrowdSim.generate_circle_crossing_human() -> CrowdSim.create_agent_attributes() ...

reset() is called in crowd_sim_dict(), which is a child class of the CrowdSim class. reset() calls CrowdSim's method generate_robot_humans() to spawn the robot **first**, then generate_random_human_position() is called where approximately n calls are made to generate_circle_crossing_human(). all human and agent positions and goals.

# config.py
## What does this file do?
All of the options are documented in config.py. These detail the following options that are available during testing/training etc.
## Why are there calculations inside config?
I've tried to put as much stuff inside config.py instead of code so that it's really **what you see is what you get** rather than having to search for some variable in some obscure file.

# Scenario Selection
Scenario lists are built using `sim.train_val_sim` during training and `sim.test_sim` during testing. Each scenario is equally weighted and randomly chosen during `CrowdSimDict.reset()`. Thus, if you want to test a specific scenario, change `sim.test_sim` to a **SINGLE ELEMENT LIST** containing your scenario. See config.py for the available scenarios

# Testing
[Kevin's paper: Social Conformity Metrics](https://hri-methods-metrics.github.io/abstract_2021/Social_Conformity_Evaluation_of_Deep_Reinforcement_Learning_Crowd_Navigation_Algorithms%20-%20Junxian%20Wang.pdf)
- To evaluate a model normally for social metrics 1-5,
    - set `test.social_metrics = True`
        - this boosts number of test episodes 500 -> 2000
        - change sim.circle_radius 6 -> 4
        - Robot behaviour: fix spawn and goal every episode instead of being random (see CrowdSim.generate_robot_humans(...))
        - Agent behaviour: depends on `sim.test_sim`

- To test social metric 6 (side preference)
    - set `sim.test_sim = x`, where x is in [side_pref_passing, side_pref_crossing, side_pref_overtaking]
        - this changes number of test episodes 500 -> 200
        - this spawns 1 agent, and 1 robot within the simulator by calling CrowdSim.generate_side_preference_scenarios()
        - Robot behaviour: fix spawn and goal every episode instead of being random (see CrowdSim.generate_robot_humans(...))
        - Agent behaviour: depends on `sim.test_sim`
# Training
Currently there are 4 available scenarios to enable during training. See [Scenario Selection](#scenario-selection)

# Phase
`self.phase = "train" or "test"` is inferred during creation of the vectorized environment. See `env.phase = "train"` in Line 70 of envs.py

#  Log Files
## Testing
A logfile will be created in data/model_name/test where the file name has the following format: `model_[checkpoint number]_test_[test name]_.log`, see the argument parser in `test.py` for full details
## Training
the training output will be available in `data/model_name/output.log`