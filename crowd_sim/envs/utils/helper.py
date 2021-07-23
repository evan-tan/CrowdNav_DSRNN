import shapely.geometry
import numpy as np
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


# https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
class VelocityRectangle:
    # how many time steps ahead to extend velocity rectangle by
    T_STEPS = 1

    def __init__(self, agent: Agent):
        self._agent = agent
        assert agent is not None
        self._create_rect()

    def _create_rect(self):
        # scaling factor for rectangle length, see Kevin's paper
        rlength = self.T_STEPS * (self._agent.vx ** 2 + self._agent.vy ** 2) ** 0.5
        rwidth = 2 * self._agent.radius
        heading = self._agent.theta

        # create rectangle at 0,0 first, bottom edge touching horizontal axis
        origin_rect = shapely.geometry.box(-rwidth / 2, 0, rwidth / 2, rlength)
        # rotate ABOUT (0,0) based on agent heading
        rot_rect = shapely.affinity.rotate(origin_rect, angle=heading, origin=(0, 0))
        # translate to get final rect
        self._vec_rect = shapely.affinity.translate(
            rot_rect, self._agent.px, self._agent.py
        )

    def intersects(self, VelocityRectangleObject):
        assert hasattr(VelocityRectangleObject, "_vec_rect")
        return self._vec_rect.intersects(VelocityRectangleObject._vec_rect)


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
