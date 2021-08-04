from typing import List

import numpy as np
import shapely.geometry
from crowd_sim.envs.utils.agent import Agent


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


def make_shapely_ellipse(radius, position, downsample=True):
    ellipse = shapely.geometry.Point(position[0], position[1]).buffer(radius)
    if downsample:
        ellipse = ellipse.simplify(0.1)
    return ellipse

# https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
class Rectangle:
    def __init__(self, width, length):
        # create vertical rectangle with center at (0,0) with heading of +90deg
        self._rect = shapely.geometry.box(
            -width / 2, -length / 2, width / 2, length / 2
        )

    def _translate(self, dx, dy):
        self._rect = shapely.affinity.translate(self._rect, dx, dy)

    def _rotate(self, angle, rot_pt="center"):
        # rot_pt = "center", "centroid" or (x,y)
        self._rect = shapely.affinity.rotate(
            self._rect, angle, origin=rot_pt, use_radians=True
        )

    def intersects(self, RectangleObject):
        assert hasattr(RectangleObject, "_rect")
        return self._rect.intersects(RectangleObject._rect)


# https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
class VelocityRectangle(Rectangle):
    # k=3, how many time steps ahead to extend velocity rectangle by
    LENGTH_SCALE = 3
    WIDTH_SCALE = 1

    def __init__(self, agent: Agent):
        self._agent = agent
        assert agent is not None

        # in agent.py [self.px, self.py, self.vx, self.vy, self.radius]
        properties = self._agent.get_observable_state_list()
        vx, vy = properties[2:4]
        radius = properties[4]
        rwidth = 2 * radius * self.WIDTH_SCALE
        rlength = self.LENGTH_SCALE * (vx ** 2 + vy ** 2) ** 0.5
        # DO NOT USE self._agent.theta as it is never updated
        # -90deg since origin_rect is ALREADY oriented at +90deg
        agent_heading = np.arctan2(vy, vx)
        dtheta = agent_heading - np.pi / 2
        x_os = self._agent.px + radius * np.cos(agent_heading)
        y_os = self._agent.py + radius * np.sin(agent_heading)

        super().__init__(rwidth, rlength)

        # make "bottom edge" of rectangle touch horizontal axis
        self._translate(0, rlength / 2)
        # rotate ABOUT (0,0) based on agent heading
        self._rotate(dtheta, (0, 0))
        # translate to get final rect
        self._translate(x_os, y_os)


class NormZoneRectangle(Rectangle):
    LENGTH_SCALE = 1
    WIDTH_SCALE = 1

    def __init__(self, agent: Agent, side="", norm="rhs"):
        self._agent = agent
        assert agent is not None
        assert "left" in side or "right" in side
        assert "lhs" in norm or "rhs" in norm

        # in agent.py [self.px, self.py, self.vx, self.vy, self.radius]
        properties = self._agent.get_observable_state_list()
        vx, vy = properties[2:4]
        radius = properties[4]
        rwidth = 2 * radius * self.WIDTH_SCALE
        rlength = self.LENGTH_SCALE * 1.2
        # DO NOT USE self._agent.theta as it is never updated
        # -90deg since origin_rect is ALREADY oriented at +90deg
        agent_heading = np.arctan2(vy, vx)
        dtheta = agent_heading - np.pi / 2
        x_os = self._agent.px + radius * np.cos(agent_heading)
        y_os = self._agent.py + radius * np.sin(agent_heading)

        super().__init__(rwidth, rlength)

        # default behaviour for rhs norm
        # make "bottom edge" of rectangle touch horizontal axis, translate to left
        if "rhs" in norm:
            if "left" in side:
                # LHS of robot
                self._translate(-rwidth / 2, rlength / 2)
            elif "right" in side:
                # RHS of robot and translated forward by 0.6m
                self._translate(rwidth / 2, rlength / 2 + 0.6)
        elif "lhs" in norm:
            if "left" in side:
                self._translate(rwidth / 2, rlength / 2 + 0.6)
            elif "right" in side:
                self._translate(-rwidth / 2, rlength / 2)

        # rotate ABOUT (0,0) based on agent heading
        self._rotate(dtheta, (0, 0))
        # translate to get final rect
        self._translate(x_os, y_os)


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
