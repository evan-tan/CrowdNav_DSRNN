import shapely.geometry
import numpy as np
from crowd_sim.envs.utils.agent import Agent
from typing import List

# wrap angle between [-180, 180]
def wrap_angle(angle):
    while angle <= -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle


def vec_norm(A, B):
    return np.linalg.norm([A[0] - B[0], A[1] - B[1]])


# smoothing function for visualizing noisy training data
def smooth_data(scalars: List[float], weight: float) -> List[float]:
    assert weight >= 0 and weight <= 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


# https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
class VelocityRectangle:
    # k=3, how many time steps ahead to extend velocity rectangle by
    T_STEPS = 3

    def __init__(self, agent: Agent):
        self._agent = agent
        assert agent is not None
        self._create_rect()

    def _create_rect(self):
        # in agent.py [self.px, self.py, self.vx, self.vy, self.radius]
        properties = self._agent.get_observable_state_list()
        vx, vy = properties[2:4]
        # scaling factor for rectangle length, see Kevin's paper
        radius = properties[4]
        rwidth = 2 * radius
        rlength = self.T_STEPS * (vx ** 2 + vy ** 2) ** 0.5
        # DO NOT USE self._agent.theta as it is never updated
        # -90deg since origin_rect is oriented at +90deg
        robot_heading = np.arctan2(vy, vx)
        box_heading = robot_heading - np.pi / 2

        # create rectangle at 0,0 first, "bottom edge" touching horizontal axis
        origin_rect = shapely.geometry.box(-rwidth / 2, 0, rwidth / 2, rlength)
        # rotate ABOUT (0,0) based on agent heading
        rot_rect = shapely.affinity.rotate(
            origin_rect, angle=box_heading, origin=(0, 0), use_radians=True
        )
        # translate to get final rect
        x_os = radius * np.cos(robot_heading)
        y_os = radius * np.sin(robot_heading)
        self._vel_rect = shapely.affinity.translate(
            rot_rect, self._agent.px + x_os, self._agent.py + y_os
        )

    def intersects(self, VelocityRectangleObject):
        assert hasattr(VelocityRectangleObject, "_vel_rect")
        return self._vel_rect.intersects(VelocityRectangleObject._vel_rect)


class SidePreference:
    def __init__(self, pos_hist_agents: dict, num_time_steps: int):
        self.history = pos_hist_agents
        self.t_steps = num_time_steps
        assert self.t_steps >= 2
        self.proximity_radius = 1.5  # metres

    def _determine_gesture(pos_r, pos_h):
        # TODO
        # translate human coordinate to the robot's position
        # calculate vectors for robot and human
        # if conditions to determine between 3 gestures [passing, overtaking, crossing]
        for k in range(len(pos_r)):
            pass

    def _slice_pos_hist(self, idx, key: str):
        # get slice of position history based on number of time steps
        pos_hist_slice = []
        offset = 0
        for _ in range(self.t_steps):
            pos_hist_slice.append(self.history.get(key)[self.t_steps * idx + offset])
            offset += 1
        return pos_hist_slice

    def calculate_side_preference(self):
        n_humans = len(self.history.keys()) - 1
        # all positions of the robot
        all_pos_r = self.history.get("robot")
        n_iter = len(all_pos_r) // self.t_steps
        for i in range(n_iter):
            set_pos_r = self._slice_pos_hist(i, "robot")
            for j in range(n_humans):
                key = "human_{}".format(j + 1)
                set_pos_h = self._slice_pos_hist(i, key)
                gesture = self._determine_gesture(set_pos_r, set_pos_h)
                # TODO take gesture that occurs most
                pass
