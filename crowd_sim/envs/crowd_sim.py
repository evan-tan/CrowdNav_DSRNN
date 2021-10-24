import logging
import random
from collections import deque

import gym
import matplotlib.legend as legend
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.policy_factory import policy_factory
from matplotlib import cm, patches
from numpy.linalg import norm

from crowd_sim.envs.utils.helper import (
    NormZoneRectangle,
    Rectangle,
    VelocityRectangle,
    check_inside_world,
    make_shapely_ellipse,
    rand_world_pt,
    vec_norm,
    wrap_angle,
)
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import Collision, Danger, Nothing, ReachGoal, Timeout
from crowd_sim.envs.utils.lidarv2 import LidarSensor
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import *


class CrowdSim(gym.Env):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.time_limit = None
        self.time_step = None
        self.robot = None  # a Robot instance representing the robot
        self.humans = None  # a list of Human instances, representing all humans in the environment
        self.global_time = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist_back = None
        self.discomfort_dist_front = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.circle_radius = None
        self.human_num = None

        self.action_space = None
        self.observation_space = None

        # limit FOV
        self.robot_fov = None
        self.human_fov = None

        self.dummy_human = None
        self.dummy_robot = None

        # seed
        self.thisSeed = None  # the seed will be set when the env is created

        # nenv
        self.nenv = None  # the number of env will be set when the env is created.
        # Because the human crossing cases are controlled by random seed, we will calculate unique random seed for each
        # parallel env.

        self.phase = None  # set the phase to be train, val or test
        self.test_case = None  # the test case ID, which will be used to calculate a seed to generate a human crossing case

        # for render
        self.render_axis = None
        self.render_figure = None
        self.render_bg = None

        self.humans = []

        self.potential = None

    def configure(self, config):
        self.config = config

        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes

        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist_back = config.reward.discomfort_dist_back
        self.discomfort_dist_front = config.reward.discomfort_dist_front
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor

        if (
            self.config.humans.policy == "orca"
            or self.config.humans.policy == "social_force"
        ):
            self.case_capacity = {
                "train": np.iinfo(np.uint32).max - 2000,
                "val": 1000,
                "test": 1000,
            }
            self.case_size = {
                "train": np.iinfo(np.uint32).max - 2000,
                "val": self.config.env.val_size,
                "test": self.config.env.test_size,
            }
            self.circle_radius = config.sim.circle_radius
            self.human_num = config.sim.human_num
            # ensure group humans are disabled if testing side pref
            self.group_human = (
                config.sim.group_human if not config.test.side_preference else False
            )

        else:
            raise NotImplementedError
        self.case_counter = {"train": 0, "test": 0, "val": 0}

        self.scenario_counter = {}
        # just to catch if you didn't update the train_config.py
        self.scenarios = {
            "train": config.sim.train_val_sim,
            "test": config.sim.test_sim,
        }
        self.scenario_counter = 0
        for _, val in self.scenarios.items():
            if type(val) is str:
                raise TypeError(
                    "config.sim.train_val_sim or config.sim.test_sim should be a list of strings. Update your config.py"
                )

        self.current_scenario = None

        logging.info("human number: {}".format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info("Circle width: {}".format(self.circle_radius))

        self.robot_fov = np.pi * config.robot.FOV
        self.human_fov = np.pi * config.humans.FOV
        logging.info("robot FOV %f", self.robot_fov)
        logging.info("humans FOV %f", self.human_fov)

        # set dummy human and dummy robot
        # dummy humans, used if any human is not in view of other agents
        self.dummy_human = Human(self.config, "humans")
        # if a human is not in view, set its state to (px = 100, py = 100, vx = 0, vy = 0, theta = 0, radius = 0)
        self.dummy_human.set(7, 7, 7, 7, 0, 0, 0)  # (7, 7, 7, 7, 0, 0, 0)
        self.dummy_human.time_step = config.env.time_step

        self.dummy_robot = Robot(self.config, "robot")
        self.dummy_robot.set(7, 7, 7, 7, 0, 0, 0)
        self.dummy_robot.time_step = config.env.time_step
        self.dummy_robot.kinematics = "holonomic"
        self.dummy_robot.policy = ORCA(config)

        # configure noise in state
        self.add_noise = config.noise.add_noise
        if self.add_noise:
            self.noise_type = config.noise.type
            self.noise_magnitude = config.noise.magnitude

        # configure randomized goal changing of humans midway through episode
        self.random_goal_changing = config.humans.random_goal_changing
        if self.random_goal_changing:
            self.goal_change_chance = config.humans.goal_change_chance

        # configure randomized goal changing of humans after reaching their respective goals
        self.end_goal_changing = config.humans.end_goal_changing
        if self.end_goal_changing:
            self.end_goal_change_chance = config.humans.end_goal_change_chance

        # configure randomized radii changing when reaching goals
        self.random_radii = config.humans.random_radii

        # configure randomized v_pref changing when reaching goals
        self.random_v_pref = config.humans.random_v_pref

        # configure randomized goal changing of humans after reaching their respective goals
        self.random_unobservability = config.humans.random_unobservability
        if self.random_unobservability:
            self.unobservable_chance = config.humans.unobservable_chance

        self.last_human_states = np.zeros((self.human_num, 5))

        # configure randomized policy changing of humans every episode
        self.random_policy_changing = config.humans.random_policy_changing

        # set robot for this envs
        rob_RL = Robot(config, "robot")
        self.set_robot(rob_RL)

        self.last_acceleration = (0, 0)
        self.last_heading = 0
        self.robot_VR = None  # robot velocity rectangle
        self.left_NZ = None  # left norm zone
        self.right_NZ = None  # right norm zone
        self.max_dist_NZ = None

        if self.render_figure:
            # grab the background on every draw
            self.render_bg = self.render_figure.canvas.copy_from_bbox(
                self.render_figure.bbox
            )
            # dummy object for matplotlib
            dummy_art = plt.Circle((0, 0), 0.5, fill=True, color="b")
            # draw the animated artist, this uses a cached renderer
            # this allows ax.draw_artist to work later
            self.render_axis.draw_artist(dummy_art)
            self.render_figure.canvas.blit(self.render_figure.bbox)

        world_size = self.config.sim.square_width
        self.world_box = Rectangle(world_size, world_size)
        t = world_size / 2
        self.wall_pts = [(-t, -t), (t, -t), (t, t), (-t, t)]
        if self.config.lidar.enable:
            self.lidar = LidarSensor(config.lidar.cfg)
        else:
            self.lidar = None
        self.lidar_end_pts = None

        # plotting
        self.circle_alpha = 0
        self.robot_history = deque(
            maxlen=int((self.config.env.time_limit - 1) / self.config.env.time_step)
        )

        return

    def set_robot(self, robot):
        raise NotImplementedError

    def generate_random_human_position(self, human_num):
        """
        Generate human position: generate start position on a circle, goal position is at the opposite side
        :param human_num:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        for i in range(human_num):
            self.humans.append(self.generate_circle_crossing_human())

    # return a static human in env
    # position: (px, py) for fixed position, or None for random position
    def generate_circle_static_obstacle(self, position=None):
        # generate a human with radius = 0.3, v_pref = 1, visible = True, and policy = orca
        human = Human(self.config, "humans")
        # For fixed position
        if position:
            px, py = position
        # For random position
        else:
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                px_noise = (np.random.random() - 0.5) * v_pref
                py_noise = (np.random.random() - 0.5) * v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist_back
                    if (
                        norm((px - agent.px, py - agent.py)) < min_dist
                        or norm((px - agent.gx, py - agent.gy)) < min_dist
                    ):
                        collide = True
                        break
                if not collide:
                    break

        # make it a static obstacle
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, px, py, 0, 0, 0, v_pref=0)
        return human

    def create_agent_attributes(self, scenario, agent):
        v_pref = 1.0 if agent.v_pref == 0 else agent.v_pref
        # add some noise to simulate all the possible cases robot could meet with human
        px_noise = (np.random.random() - 0.5) * v_pref
        py_noise = (np.random.random() - 0.5) * v_pref
        heading = 0

        if "circle_crossing" == scenario:
            angle = np.random.random() * np.pi * 2
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            gx = -px
            gy = -py
        elif "square_crossing" == scenario:
            # rand_world_pt generates random point in (-10,10)
            # based on config.sim.square_width
            px = rand_world_pt(self.config) * 0.4 + px_noise
            py = rand_world_pt(self.config) * 0.4 + py_noise
            gx = rand_world_pt(self.config) * 0.4 + px_noise
            gy = rand_world_pt(self.config) * 0.4 + py_noise
        elif "parallel_traffic" == scenario:
            # whether or not to flip sides
            sign = 1 if np.random.random() >= 0.5 else -1
            px = rand_world_pt(self.config) * 0.4 + px_noise
            # (1,4) or (-1,-4)
            py = sign * (np.random.random() * 3 + 1 + py_noise)
            gx = px
            gy = -py
        elif "perpendicular_traffic" == scenario:
            sign = 1 if np.random.random() >= 0.5 else -1
            # (1,4) or (-1,-4)
            px = sign * (np.random.random() * 3 + 1 + px_noise)
            gx = -px
            py = rand_world_pt(self.config) * 0.4 + py_noise
            gy = py
        elif scenario in ["side_pref_passing", "side_pref_overtaking"]:
            min_x = -(self.robot.radius + agent.radius)
            max_x = -min_x
            human_x = (max_x - min_x) * np.random.random() + min_x
            px, gx = human_x, human_x
            if "side_pref_passing" == scenario:
                py, gy = self.circle_radius, -self.circle_radius
                heading = -np.pi / 2
            elif "side_pref_overtaking" == scenario:
                offset = 2
                py = -self.circle_radius + offset
                gy = self.circle_radius + offset
                # flip to opposite side and offset
                heading = np.pi / 2
                v_pref = 0.3
        elif scenario == "side_pref_crossing":
            min_x = -(self.circle_radius + self.robot.radius + agent.radius)
            max_x = -(self.circle_radius - self.robot.radius - agent.radius)
            human_x = (max_x - min_x) * np.random.random() + min_x
            px, gx = human_x, -human_x
            py, gy = 0, 0
        # generate spawn and goal positions
        return px, py, gx, gy, heading, v_pref

    def generate_circle_crossing_human(self):
        human = Human(self.config, "humans")
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            # create spawn/goal based on scenario
            px, py, gx, gy, heading, vel = self.create_agent_attributes(
                self.current_scenario, human
            )

            collide = False
            # for Group environment
            if self.group_human:
                collide = self.check_collision_group((px, py), human.radius)

            # for FoV environment
            else:
                for i, agent in enumerate([self.robot] + self.humans):
                    # keep human at least 3 meters away from robot
                    if self.robot.kinematics == "unicycle" and i == 0:
                        min_dist = (
                            self.circle_radius / 2
                        )  # Todo: if circle_radius <= 4, it will get stuck here
                    else:
                        min_dist = (
                            human.radius + agent.radius + self.discomfort_dist_back
                        )
                    if vec_norm([px, py], [agent.px, agent.py]) < min_dist:
                        collide = True
                        break
            if not collide:
                break

        human.set(px, py, gx, gy, 0, 0, heading, v_pref=vel)
        return human

    # add noise according to env.config to state
    def apply_noise(self, ob):
        if isinstance(ob[0], ObservableState):
            for i in range(len(ob)):
                if self.noise_type == "uniform":
                    noise = np.random.uniform(
                        -self.noise_magnitude, self.noise_magnitude, 5
                    )
                elif self.noise_type == "gaussian":
                    noise = np.random.normal(size=5)
                else:
                    print("noise type not defined")
                ob[i].px = ob[i].px + noise[0]
                ob[i].py = ob[i].px + noise[1]
                ob[i].vx = ob[i].px + noise[2]
                ob[i].vy = ob[i].px + noise[3]
                ob[i].radius = ob[i].px + noise[4]
            return ob
        else:
            if self.noise_type == "uniform":
                noise = np.random.uniform(
                    -self.noise_magnitude, self.noise_magnitude, len(ob)
                )
            elif self.noise_type == "gaussian":
                noise = np.random.normal(size=len(ob))
            else:
                print("noise type not defined")
                noise = [0] * len(ob)

            return ob + noise

    # update the robot belief of human states
    # if a human is visible, its state is updated to its current ground truth state
    # else we assume it keeps going in a straight line with last observed velocity
    def update_last_human_states(self, human_visibility, reset):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        # keep the order of 5 humans at each timestep
        for i in range(self.human_num):
            if human_visibility[i]:
                humanS = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i, :] = humanS

            else:
                if reset:
                    humanS = np.array([15.0, 15.0, 0.0, 0.0, 0.3])
                    self.last_human_states[i, :] = humanS

                else:
                    px, py, vx, vy, r = self.last_human_states[i, :]
                    # Plan A: linear approximation of human's next position
                    px = px + vx * self.time_step
                    py = py + vy * self.time_step
                    self.last_human_states[i, :] = np.array([px, py, vx, vy, r])

                    # Plan B: assume the human doesn't move, use last observation
                    # self.last_human_states[i, :] = np.array([px, py, 0., 0., r])

    # return the ground truth locations of all humans
    def get_true_human_states(self):
        true_human_states = np.zeros((self.human_num, 2))
        for i in range(self.human_num):
            humanS = np.array(self.humans[i].get_observable_state_list())
            true_human_states[i, :] = humanS[:2]
        return true_human_states

    def randomize_human_policies(self):
        """
        Randomize the moving humans' policies to be either orca or social force
        """
        for human in self.humans:
            if not human.isObstacle:
                new_policy = random.choice(["orca", "social_force"])
                new_policy = policy_factory[new_policy]()
                human.set_policy(new_policy)

    # Generates group of circum_num humans in a circle formation at a random viable location
    def generate_circle_group_obstacle(self, circum_num):
        group_circumference = self.config.humans.radius * 2 * circum_num
        # print("group circum: ", group_circumference)
        group_radius = group_circumference / (2 * np.pi)
        # print("group radius: ", group_radius)
        while True:
            rand_cen_x = np.random.uniform(-3, 3)
            rand_cen_y = np.random.uniform(-3, 3)
            success = True
            for i, group in enumerate(self.circle_groups):
                # print(i)
                dist_between_groups = np.sqrt(
                    (rand_cen_x - group[1]) ** 2 + (rand_cen_y - group[2]) ** 2
                )
                sum_radius = group_radius + group[0] + 2 * self.config.humans.radius
                if dist_between_groups < sum_radius:
                    success = False
                    break
            if success:
                # print("------------\nsuccessfully found valid x: ", rand_cen_x, " y: ", rand_cen_y)
                break
        self.circle_groups.append((group_radius, rand_cen_x, rand_cen_y))

        # print("current groups:")
        # for i in self.circle_groups:
        #     print(i)

        arc = 2 * np.pi / circum_num
        for i in range(circum_num):
            angle = arc * i
            curr_x = rand_cen_x + group_radius * np.cos(angle)
            curr_y = rand_cen_y + group_radius * np.sin(angle)
            point = (curr_x, curr_y)
            # print("adding circle point: ", point)
            curr_human = self.generate_circle_static_obstacle(point)
            curr_human.isObstacle = True
            self.humans.append(curr_human)

        return

    # given an agent's xy location and radius, check whether it collides with all other humans
    # including the circle group and moving humans
    # return True if collision, False if no collision
    def check_collision_group(self, pos, radius):
        # check circle groups
        for r, x, y in self.circle_groups:
            if (
                np.linalg.norm([pos[0] - x, pos[1] - y]) <= r + radius + 2 * 0.5
            ):  # use 0.5 because it's the max radius of human
                return True

        # check moving humans
        for human in self.humans:
            if human.isObstacle:
                pass
            else:
                if (
                    np.linalg.norm([pos[0] - human.px, pos[1] - human.py])
                    <= human.radius + radius
                ):
                    return True
        return False

    # check collision between robot goal position and circle groups
    def check_collision_group_goal(self, pos, radius):
        # print('check goal', len(self.circle_groups))
        collision = False
        # check circle groups
        for r, x, y in self.circle_groups:
            # print(np.linalg.norm([pos[0] - x, pos[1] - y]), r + radius + 4 * 0.5)
            if (
                np.linalg.norm([pos[0] - x, pos[1] - y]) <= r + radius + 4 * 0.5
            ):  # use 0.5 because it's the max radius of human
                collision = True
        return collision

    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        if human_num is None:
            human_num = self.human_num
        # for Group environment
        if self.group_human:
            # set the robot in a dummy far away location to avoid collision with humans
            if self.config.test.social_metrics:
                self.robot.set(
                    0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2
                )
            else:
                self.robot.set(10, 10, 10, 10, 0, 0, np.pi / 2)

            # generate humans
            self.circle_groups = []
            humans_left = human_num

            while humans_left > 0:
                # print("****************\nhumans left: ", humans_left)
                if humans_left <= 4:
                    if phase in ["train", "val"]:
                        self.generate_random_human_position(human_num=humans_left)
                    else:
                        self.generate_random_human_position(human_num=humans_left)
                    humans_left = 0
                else:
                    if humans_left < 10:
                        max_rand = humans_left
                    else:
                        max_rand = 10
                    # print("randint from 4 to ", max_rand)
                    circum_num = np.random.randint(4, max_rand)
                    # print("circum num: ", circum_num)
                    self.generate_circle_group_obstacle(circum_num)
                    humans_left -= circum_num

            # randomize starting position and goal position while keeping the distance of goal to be > 6
            # set the robot on a circle with radius 5.5 randomly
            rand_angle = np.random.uniform(0, np.pi * 2)
            # print('rand angle:', rand_angle)
            increment_angle = 0.0
            while True:
                px_r = np.cos(rand_angle + increment_angle) * 5.5
                py_r = np.sin(rand_angle + increment_angle) * 5.5
                # check whether the initial px and py collides with any human
                collision = self.check_collision_group((px_r, py_r), self.robot.radius)
                # if the robot goal does not fall into any human groups, the goal is okay, otherwise keep generating the goal
                if not collision:
                    # print('initial pos angle:', rand_angle+increment_angle)
                    break
                increment_angle = increment_angle + 0.2

            increment_angle = (
                increment_angle + np.pi
            )  # start at opposite side of the circle
            while True:
                gx = np.cos(rand_angle + increment_angle) * 5.5
                gy = np.sin(rand_angle + increment_angle) * 5.5
                # check whether the goal is inside the human groups
                # check whether the initial px and py collides with any human
                collision = self.check_collision_group_goal((gx, gy), self.robot.radius)
                # if the robot goal does not fall into any human groups, the goal is okay, otherwise keep generating the goal
                if not collision:
                    # print('goal pos angle:', rand_angle + increment_angle)
                    break
                increment_angle = increment_angle + 0.2

            self.robot.set(px_r, py_r, gx, gy, 0, 0, np.pi / 2)

        # for FoV environment
        else:
            if self.robot.kinematics == "unicycle":
                angle = np.random.uniform(0, np.pi * 2)
                px = self.circle_radius * np.cos(angle)
                py = self.circle_radius * np.sin(angle)
                while True:
                    gx, gy = np.random.uniform(
                        -self.circle_radius, self.circle_radius, 2
                    )
                    if np.linalg.norm([px - gx, py - gy]) >= 6:  # 1 was 6
                        break
                self.robot.set(
                    px, py, gx, gy, 0, 0, np.random.uniform(0, 2 * np.pi)
                )  # randomize init orientation

            # randomize starting position and goal position
            else:
                if self.config.test.social_metrics or self.config.test.side_preference:
                    self.robot.set(
                        0,
                        -self.circle_radius,
                        0,
                        self.circle_radius,
                        0,
                        0,
                        np.pi / 2,
                    )
                else:
                    while True:
                        px, py, gx, gy = np.random.uniform(
                            -self.circle_radius, self.circle_radius, 4
                        )
                        # ensure minimum distance of 6m
                        if np.linalg.norm([px - gx, py - gy]) >= 6:
                            break
                    self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)

            # generate humans
            self.generate_random_human_position(human_num=human_num)

    def reset(self, phase="train", test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case

        if self.robot is None:
            raise AttributeError("robot has to be set!")
        assert phase in ["train", "val", "test"]
        if test_case is not None:
            self.case_counter[
                phase
            ] = test_case  # test case is passed in to calculate specific seed to generate case
        self.global_time = 0

        self.humans = []
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {
            "train": self.case_capacity["val"] + self.case_capacity["test"],
            "val": 0,
            "test": self.case_capacity["val"],
        }

        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)

        self.generate_robot_humans(phase)

        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (
            self.case_counter[phase] + int(1 * self.nenv)
        ) % self.case_size[phase]

        # get current observation
        ob = self.generate_ob(reset=True)

        # initialize potential
        self.potential = -abs(
            np.linalg.norm(
                np.array([self.robot.px, self.robot.py])
                - np.array([self.robot.gx, self.robot.gy])
            )
        )

        return ob

    # Update the humans' end goals in the environment
    # Produces valid end goals for each human
    def update_human_goals_randomly(self):
        # Update humans' goals randomly
        for human in self.humans:
            if human.isObstacle or human.v_pref == 0:
                continue
            if np.random.random() <= self.goal_change_chance:
                if not self.group_human:  # to improve the runtime
                    humans_copy = []
                    for h in self.humans:
                        if h != human:
                            humans_copy.append(h)

                # Produce valid goal for human in case of circle setting
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases robot could meet with human
                    v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                    gx_noise = (np.random.random() - 0.5) * v_pref
                    gy_noise = (np.random.random() - 0.5) * v_pref
                    gx = self.circle_radius * np.cos(angle) + gx_noise
                    gy = self.circle_radius * np.sin(angle) + gy_noise
                    collide = False

                    if self.group_human:
                        collide = self.check_collision_group((gx, gy), human.radius)
                    else:
                        for agent in [self.robot] + humans_copy:
                            min_dist = (
                                human.radius + agent.radius + self.discomfort_dist_back
                            )
                            if (
                                norm((gx - agent.px, gy - agent.py)) < min_dist
                                or norm((gx - agent.gx, gy - agent.gy)) < min_dist
                            ):
                                collide = True
                                break
                    if not collide:
                        break

                # Give human new goal
                human.gx = gx
                human.gy = gy
        return

    # Update the specified human's end goals in the environment based on scenario
    def update_human_goal(self, human):

        # Update human's goals randomly
        if np.random.random() <= self.end_goal_change_chance:
            if not self.group_human:
                humans_copy = []
                for h in self.humans:
                    if h != human:
                        humans_copy.append(h)

            # Update human's radius now that it's reached goal
            if self.random_radii:
                human.radius += np.random.uniform(-0.1, 0.1)

            # Update human's v_pref now that it's reached goal
            if self.random_v_pref:
                human.v_pref += np.random.uniform(-0.1, 0.1)

            while True:
                _, _, gx, gy, _, _ = self.create_agent_attributes(
                    self.current_scenario, human
                )
                collide = False
                if self.group_human:
                    collide = self.check_collision_group((gx, gy), human.radius)
                else:
                    for agent in [self.robot] + humans_copy:
                        min_dist = (
                            human.radius + agent.radius + self.discomfort_dist_back
                        )
                        if (
                            norm((gx - agent.px, gy - agent.py)) < min_dist
                            or norm((gx - agent.gx, gy - agent.gy)) < min_dist
                        ):
                            collide = True
                            break
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gy
        return

    # Caculate whether agent2 is in agent1's FOV
    # Not the same as whether agent1 is in agent2's FOV!!!!
    # arguments:
    # state1, state2: can be agent instance OR state instance
    # robot1: is True if state1 is robot, else is False
    # return value:
    # return True if state2 is visible to state1, else return False
    def detect_visible(self, state1, state2, robot1=False, custom_fov=None):
        if self.robot.kinematics == "holonomic":
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov
            else:
                fov = self.human_fov

        if np.abs(offset) <= fov / 2:
            return True
        else:
            return False

    # for robot:
    # return only visible humans to robot and number of visible humans and visible humans' ids (0 to 4)
    def get_num_human_in_fov(self):
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        for i in range(self.human_num):
            visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids

    # convert an np array with length = 34 to a JointState object
    def array_to_jointstate(self, obs_list):
        fullstate = FullState(
            obs_list[0],
            obs_list[1],
            obs_list[2],
            obs_list[3],
            obs_list[4],
            obs_list[5],
            obs_list[6],
            obs_list[7],
            obs_list[8],
        )

        observable_states = []
        for k in range(self.human_num):
            idx = 9 + k * 5
            observable_states.append(
                ObservableState(
                    obs_list[idx],
                    obs_list[idx + 1],
                    obs_list[idx + 2],
                    obs_list[idx + 3],
                    obs_list[idx + 4],
                )
            )
        state = JointState(fullstate, observable_states)
        return state

    def last_human_states_obj(self):
        """
        convert self.last_human_states to a list of observable state objects for old algorithms to use
        """
        humans = []
        for i in range(self.human_num):
            h = ObservableState(*self.last_human_states[i])
            humans.append(h)
        return humans

    # find R(s, a)
    def calc_reward(self, action):
        # collision detection
        dmin = float("inf")

        danger_dists = []
        collision = False
        step_info = dict()
        self.robot_VR = VelocityRectangle(self.robot)

        if self.config.reward.norm_zones:
            social_norm = self.config.reward.norm_zone_side
            self.left_NZ = NormZoneRectangle(self.robot, side="left", norm=social_norm)
            self.right_NZ = NormZoneRectangle(
                self.robot, side="right", norm=social_norm
            )

            if self.max_dist_NZ is None:
                distances = []
                for idx, zone in enumerate([self.left_NZ, self.right_NZ]):
                    for point in zone._rect.exterior.coords:
                        distances.append(
                            vec_norm([self.robot.px, self.robot.py], point)
                        )
                self.max_dist_NZ = max(distances)

        vec_rect_violations = 0  # social zone violations
        aggregate_nav_time = 0
        norm_zone_violations = 0
        robot_pos = self.robot.get_position()

        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** 0.5 - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist_back:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

            human_pos = human.get_position()
            # check if norm zones violated
            if self.config.reward.norm_zones:
                if vec_norm(human_pos, robot_pos) <= self.max_dist_NZ:
                    h_ellipse = make_shapely_ellipse(human.radius, human_pos)
                    for zone in [self.left_NZ, self.right_NZ]:
                        if h_ellipse.intersects(zone._rect):
                            norm_zone_violations += 1

            # SOCIAL METRIC 2
            human_VR = VelocityRectangle(human)
            if self.robot_VR is not None:
                if self.robot_VR.intersects(human_VR):
                    vec_rect_violations += 1
            # SOCIAL METRIC 3
            if not human.reached_destination():
                aggregate_nav_time += 1

        if not self.robot.reached_destination():
            aggregate_nav_time += 1
        step_info["aggregate_nav_time"] = aggregate_nav_time
        step_info["path_violation"] = vec_rect_violations

        # SOCIAL METRIC 6
        if self.config.test.side_preference:
            side_preference = {"left": 0, "right": 0}
            scenario = self.current_scenario
            end_pos_r = self.robot.compute_position(action, self.time_step)
            # self.humans should only contain 1 human
            h = self.humans[0]
            # check if robot y within human's radius
            if end_pos_r[1] <= (h.py + h.radius) and end_pos_r[1] >= (h.py - h.radius):
                if end_pos_r[0] < h.px:
                    side_preference["left"] = 1
                else:
                    side_preference["right"] = 1
            step_info[scenario] = side_preference
            step_info["separation"] = vec_norm(
                h.get_position(), self.robot.get_position()
            )

        # check if reaching the goal
        reaching_goal = self.robot.reached_destination()

        # SOCIAL METRIC 1
        if dmin < self.config.social.min_personal_space:
            step_info["personal_violation"] = 1
        else:
            step_info["personal_violation"] = 0
        # SOCIAL METRIC 4
        # calculate step jerk cost
        ax = action.vx - self.robot.vx
        ay = action.vy - self.robot.vy
        d_ax = ax - self.last_acceleration[0]
        d_ay = ay - self.last_acceleration[1]
        jerk_cost = d_ax ** 2 + d_ay ** 2
        self.last_acceleration = (ax, ay)

        step_info["jerk_cost"] = jerk_cost
        step_info["dist_to_goal"] = vec_norm(
            self.robot.get_position(), self.robot.get_goal_position()
        )

        robot_ellipse = make_shapely_ellipse(
            self.robot.radius, self.robot.get_position()
        )
        # use this boolean to check if robot is inside world
        inside_world = check_inside_world(robot_ellipse, self.wall_pts)

        # SOCIAL METRIC 5
        speed = (action.vx ** 2 + action.vy ** 2) ** 0.5
        if speed > self.config.social.max_walking_speed:
            step_info["speed_violation"] = 1
        else:
            step_info["speed_violation"] = 0

        time_discount_factor = (self.time_limit - self.global_time) / self.time_limit
        step_info["scenario"] = self.current_scenario

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            step_info["event"] = Timeout()
        # additional check to end episode if robot veers off too far
        elif collision or not inside_world:
            reward = self.collision_penalty
            done = True
            step_info["event"] = Collision()
        elif reaching_goal:
            reward = self.success_reward
            if self.config.reward.time_factor:
                reward *= time_discount_factor

            done = True
            step_info["event"] = ReachGoal()

        elif dmin < self.discomfort_dist_back:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            # print(dmin)
            reward = (dmin - self.discomfort_dist_back) * self.discomfort_penalty_factor
            done = False
            step_info["event"] = Danger(dmin)

        else:
            # potential reward
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py])
                - np.array(self.robot.get_goal_position())
            )
            if self.config.reward.potential_based:
                reward = self.config.reward.potential_factor * (
                    -abs(potential_cur) - self.potential
                )
                self.potential = -abs(potential_cur)
            elif self.config.reward.exponential:
                reward = self.config.reward.exp_factor * (
                    1 - (potential_cur / self.config.reward.exp_denom) ** 0.4
                )

            if self.config.reward.norm_zones:
                reward += self.config.reward.norm_zone_penalty * norm_zone_violations

            done = False
            step_info["event"] = Nothing()

        # if the robot is near collision/arrival, it should be able to turn a large angle
        if self.robot.kinematics == "unicycle":
            # add a rotational penalty
            # if action.r is w, factor = -0.02 if w in [-1.5, 1.5], factor = -0.045 if w in [-1, 1];
            # if action.r is delta theta, factor = -2 if r in [-0.15, 0.15], factor = -4.5 if r in [-0.1, 0.1]
            r_spin = -2 * action.r ** 2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.0
            # print(reward, r_spin, r_back)
            reward = reward + r_spin + r_back

        return reward, done, step_info

    # compute the observation
    def generate_ob(self, reset):
        (
            visible_human_states,
            num_visible_humans,
            human_visibility,
        ) = self.get_num_human_in_fov()
        self.update_last_human_states(human_visibility, reset=reset)
        if self.robot.policy.name in ["lstm_ppo", "srnn"]:
            ob = [num_visible_humans]
            # append robot's state
            robotS = np.array(self.robot.get_full_state_list())
            ob.extend(list(robotS))

            ob.extend(list(np.ravel(self.last_human_states)))
            ob = np.array(ob)

        else:  # for orca and sf
            ob = self.last_human_states_obj()

        if self.add_noise:
            ob = self.apply_noise(ob)

        return ob

    def get_human_actions(self):
        # step all humans
        human_actions = []  # a list of all humans' actions
        for i, human in enumerate(self.humans):
            # observation for humans is always coordinates
            ob = []
            for other_human in self.humans:
                if other_human != human:
                    # Chance for one human to be blind to some other humans
                    if self.random_unobservability and i == 0:
                        if (
                            np.random.random() <= self.unobservable_chance
                            or not self.detect_visible(human, other_human)
                        ):
                            ob.append(self.dummy_human.get_observable_state())
                        else:
                            ob.append(other_human.get_observable_state())
                    # Else detectable humans are always observable to each other
                    elif self.detect_visible(human, other_human):
                        ob.append(other_human.get_observable_state())
                    else:
                        ob.append(self.dummy_human.get_observable_state())

            if self.robot.visible:
                # Chance for one human to be blind to robot
                if self.random_unobservability and i == 0:
                    if (
                        np.random.random() <= self.unobservable_chance
                        or not self.detect_visible(human, self.robot)
                    ):
                        ob += [self.dummy_robot.get_observable_state()]
                    else:
                        ob += [self.robot.get_observable_state()]
                # Else human will always see visible robots
                elif self.detect_visible(human, self.robot):
                    ob += [self.robot.get_observable_state()]
                else:
                    ob += [self.dummy_robot.get_observable_state()]

            human_actions.append(human.act(ob))
        return human_actions

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """

        # clip the action to obey robot's constraint
        action = self.robot.policy.clip_action(action, self.robot.v_pref)

        # step all humans
        human_actions = self.get_human_actions()

        # compute reward and episode info
        reward, done, step_info = self.calc_reward(action)
        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step  # max episode length=time_limit/time_step

        ##### compute_ob goes here!!!!!
        ob = self.generate_ob(reset=False)

        if self.robot.policy.name in ["srnn"]:
            info = {"info": step_info}
        else:  # for orca and sf
            info = step_info

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if (
                    not human.isObstacle
                    and human.v_pref != 0
                    and norm((human.gx - human.px, human.gy - human.py)) < human.radius
                ):
                    self.update_human_goal(human)

        return ob, reward, done, info

    def render(self, mode="human"):
        # reset the background back in the canvas state, screen unchanged
        self.render_figure.canvas.restore_region(self.render_bg)

        plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

        robot_color = "yellow"
        goal_color = "red"
        arrow_color = "red"
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array(
                [
                    [np.cos(ang), -np.sin(ang), 0],
                    [np.sin(ang), np.cos(ang), 0],
                    [0, 0, 1],
                ]
            )
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        ax = self.render_axis
        # NOTE: sets are NOT ordered
        unordered_artists = set()  # use set to speed up animation,
        ordered_artists = list()  # use for drawings that have to be in foreground

        ###### START LIDAR DRAWING #####
        from crowd_sim.envs.utils.lidarv2 import rotate, unsqueeze

        if self.lidar_end_pts is not None and self.config.lidar.viz:
            lidar_poly = patches.Polygon(self.lidar_end_pts, fill=False)
            ax.add_artist(lidar_poly)
            ordered_artists.append(lidar_poly)

            # for i in range(self.lidar_end_pts.shape[0]):
            #     # color first line different
            #     color = "r" if i == 0 else "k"
            #     x_data = (self.lidar.sensor_pos[0], self.lidar_end_pts[i, 0])
            #     y_data = (self.lidar.sensor_pos[1], self.lidar_end_pts[i, 1])
            #     lidar_beam = mlines.Line2D(x_data, y_data, color=color, alpha=0.5)
            #     ax.add_artist(lidar_beam)
            #     unordered_artists.add(lidar_beam)

        ##### END LIDAR DRAWING #####

        # DRAW BOX
        walls_coords = self.world_box._rect.exterior.coords
        walls = patches.Polygon(xy=walls_coords, fill=False)
        ax.add_artist(walls)
        unordered_artists.add(walls)

        ###### START ROBOT DRAWING ######

        # if self.robot_VR is not None:
        #     polygon = patches.Polygon(
        #         xy=self.robot_VR._rect.exterior.coords, fill=False
        #     )
        #     polygon.set_animated(False)
        #     ax.add_artist(polygon)
        #     unordered_artists.add(polygon)

        # if self.left_NZ is not None and self.right_NZ is not None:
        #     for i, NZ in enumerate([self.left_NZ, self.right_NZ]):
        #         if i == 0:
        #             color = 'b'
        #         else:
        #             color = 'r'
        #         polygon = patches.Polygon(
        #             xy=NZ._rect.exterior.coords, fill=False, color=color
        #         )
        #         polygon.set_animated(False)
        #         ax.add_artist(polygon)
        #         unordered_artists.add(polygon)

        # add goal
        goal = mlines.Line2D(
            [self.robot.gx],
            [self.robot.gy],
            color=goal_color,
            marker="*",
            linestyle="None",
            markersize=15,
            label="Goal",
        )
        goal.set_animated(False)
        ax.add_artist(goal)
        unordered_artists.add(goal)

        # add robot
        robotX, robotY = self.robot.get_position()
        robot = plt.Circle(
            (robotX, robotY), self.robot.radius, fill=True, color=robot_color
        )

        # TEMPORAL DRAWING
        temporal_graph = False
        if temporal_graph and int(self.global_time % 1) == 0:
            self.robot_history.append((robotX, robotY))
            if len(self.robot_history) > 0:
                alpha = np.linspace(0.05, 0.6, len(self.robot_history))
                for i, pos in enumerate(self.robot_history):
                    tmp_circle = plt.Circle(
                        pos, self.robot.radius, fill=True, color="k", alpha=alpha[i]
                    )
                    tmp_circle.set_animated(False)
                    ax.add_artist(tmp_circle)
                    ordered_artists.append(tmp_circle)

        robot.set_animated(True)
        ax.add_artist(robot)
        ordered_artists.append(robot)

        blit_legend = plt.legend([robot, goal], ["Robot", "Goal"], fontsize=16)
        blit_legend.set_animated(False)
        ax.add_artist(blit_legend)
        ordered_artists.append(blit_legend)

        robot_speed = mtext.Text(
            self.robot.get_position()[0],
            self.robot.get_position()[1] + self.robot.radius,
            f"{vec_norm(self.robot.get_velocity(), [0, 0]):.2f}",
            ha="center",
            fontsize=9,
        )
        robot_speed.set_animated(True)
        ax.add_artist(robot_speed)
        ordered_artists.append(robot_speed)

        if "unicycle" in self.robot.kinematics:
            robot_theta = self.robot.theta
        else:
            robot_theta = np.arctan2(self.robot.vy, self.robot.vx)
        arrow_start = self.robot.get_position()
        arrow_end = [
            self.robot.px + self.robot.radius * np.cos(robot_theta),
            self.robot.py + self.robot.radius * np.sin(robot_theta),
        ]
        arrow_patch = patches.FancyArrowPatch(
            posA=arrow_start, posB=arrow_end, color=arrow_color, arrowstyle=arrow_style
        )

        arrow_patch.set_animated(True)
        ax.add_artist(arrow_patch)
        ordered_artists.append(arrow_patch)

        # draw FOV for the robot
        # add robot FOV
        if self.robot_fov < np.pi * 2:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle="--")
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle="--")

            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + self.robot.radius * np.cos(robot_theta)
            endPointY = robotY + self.robot.radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(
                FOVAng,
                [endPointX - startPointX, endPointY - startPointY],
                20.0 / self.robot.radius,
            )
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(
                -FOVAng,
                [endPointX - startPointX, endPointY - startPointY],
                20.0 / self.robot.radius,
            )
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            FOVLine1.set_animated(True)
            ax.add_artist(FOVLine1)
            unordered_artists.add(FOVLine1)

            FOVLine2.set_animated(True)
            ax.add_artist(FOVLine2)
            unordered_artists.add(FOVLine2)

        ###### END DRAWING ROBOT ######
        ###### START DRAWING HUMANS ######
        for idx, human in enumerate(self.humans):
            if "unicycle" in human.kinematics:
                theta = human.theta
            else:
                theta = np.arctan2(human.vy, human.vx)

            arrow_start = human.get_position()
            arrow_end = [
                human.px + human.radius * np.cos(theta),
                human.py + human.radius * np.sin(theta),
            ]
            arrow_patch = patches.FancyArrowPatch(
                posA=arrow_start,
                posB=arrow_end,
                color=arrow_color,
                arrowstyle=arrow_style,
            )

            human_circle = plt.Circle(human.get_position(), human.radius, fill=False)

            speed = vec_norm(human.get_velocity(), [0, 0])
            human_speed_text = None
            if speed > 0:
                arrow_patch.set_animated(True)
                human_circle.set_animated(True)

                human_speed_text = mtext.Text(
                    human.get_position()[0],
                    human.get_position()[1] + human.radius,
                    f"{speed:.2f}",
                    ha="center",
                    fontsize=9,
                )
                human_speed_text.set_animated(True)

                ax.add_artist(human_speed_text)
                unordered_artists.add(human_speed_text)
            else:
                arrow_patch.set_animated(False)
                human_circle.set_animated(False)

            # green: visible; red: invisible
            if self.detect_visible(self.robot, human, robot1=True):
                human_circle.set_color(c="g")
            else:
                human_circle.set_color(c="r")

            ax.add_artist(arrow_patch)
            unordered_artists.add(arrow_patch)

            ax.add_artist(human_circle)
            unordered_artists.add(human_circle)

        ###### END DRAWING HUMANS ######

        for art in unordered_artists:
            self.render_axis.draw_artist(art)
        for other_art in ordered_artists:
            self.render_axis.draw_artist(other_art)

        # copy the image to the GUI state, but screen might not be changed yet
        self.render_figure.canvas.blit(self.render_figure.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        self.render_figure.canvas.flush_events()
