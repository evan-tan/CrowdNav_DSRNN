from collections import deque
from typing import List

import matplotlib.lines as mlines
import numpy as np
import shapely.geometry
from crowd_sim.envs.utils.agent import Agent


def unsqueeze(arr: np.ndarray, dim: int) -> np.ndarray:
    """Wrapper function for torch.unsqueeze() functionality in NumPy"""
    return np.expand_dims(arr, axis=dim)


def create_agents_arr(agent_xyr: np.ndarray, n_pts: int) -> np.ndarray:
    """Create all agent points in the WORLD FRAME

    :param agent_xyr: All agent obstacle attributes, ASSUMED to be in WORLD FRAME, shape=(n_agents, 3)
    :type agent_xyr: np.ndarray
    :param n_pts: number of points to represent agent obstacle polygon
    :type n_pts: int
    :return: Array of xy points for each agent polygon, shape=(n_agent,2, n_pts)
    :rtype: np.ndarray
    """
    # points to represent polygon
    poly_angles = np.linspace(0, 2 * np.pi, n_pts)
    # (n_agents,) -> (n_agents,1)
    radii = unsqueeze(agent_xyr[:, 2], dim=1)
    world_x = unsqueeze(agent_xyr[:, 0], dim=1)
    world_y = unsqueeze(agent_xyr[:, 1], dim=1)

    # (n_agents,2,n_pts)
    agent_arr = np.zeros((agent_xyr.shape[0], 2, n_pts))
    # broadcasting, position in WORLD FRAME
    agent_arr[:, 0] = world_x + radii * np.cos(poly_angles)
    agent_arr[:, 1] = world_y + radii * np.sin(poly_angles)
    return agent_arr


def create_events_dict(config):
    """Create nested events dictionary to count terminal states for each different scenario from config.py

    :param config: CrowdSim config
    :type config: Config
    :return: dictionary containing all events as keys
    :rtype: dict of dicts
    """
    # breakdown cases according to scenarios
    num_events = {
        "success": {},
        "collision": {},
        "timeout": {},
    }
    # create a set of unique keys
    scenarios = set(config.sim.train_val_sim).union(set(config.sim.test_sim))
    for key in num_events.keys():
        num_events[key]["total"] = 0
        for scenario in scenarios:
            num_events[key][scenario] = 0
    return num_events


def depth(d):
    """Check dictionary depth"""
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0


def log_events_dict(events_dict, logger):
    """Log events dictionary to logfile during testing

    :param events_dict: events dictionary with depth 2
    :type events_dict: dict of dicts
    :param logger: logging module or logger object
    :type logger: logging object
    """
    assert depth(events_dict) == 2
    for k in events_dict.keys():
        logger.info("")
        logger.info(f"{k.upper()} CASES: ")
        for scenario, count in events_dict[k].items():
            logger.info(f"{scenario}: {count}")


def rand_world_pt(config):
    """Generate random world point based on square_width"""
    # this assumes world is centered at (0,0)
    return (np.random.random() - 0.5) * config.sim.square_width / 2


def wrap_angle(angle):
    """Wrap angle between [-180, 180]"""
    while angle <= -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle


def vec_norm(A, B):
    """Small wrapper to calculate L2 norm for 2 points"""
    return np.linalg.norm([A[0] - B[0], A[1] - B[1]])


# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth_data(scalars: List[float], weight: float) -> List[float]:
    """Tensorboard smoothing function to smooth noisy training data

    :param scalars: data points to smooth
    :type scalars: List[float]
    :param weight: Exponential Moving Average weight in 0-1
    :type weight: float
    :return: smoothed data points
    :rtype: List[float]
    """
    assert weight >= 0 and weight <= 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def make_mpl_line(x, y):
    """Wrapper to create matplotlib line"""
    return mlines.Line2D(
        x, y, marker="o", markersize=2, color="r", solid_capstyle="round"
    )


def make_shapely_polygon(pts, downsample=False):
    """Wrapper to create Shapely polygon"""
    # ensure wrap around
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    poly = shapely.geometry.Polygon(pts)
    if downsample:
        poly = poly.simplify(0.1)
    return poly


def make_shapely_ellipse(radius, position, downsample=False):
    """Wrapper to create Shapely ellipse"""
    ellipse = shapely.geometry.Point(position[0], position[1]).buffer(radius)
    if downsample:
        ellipse = ellipse.simplify(0.1)
    return ellipse


# https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
class Rectangle:
    """Rectangle base class.
    Creates rectangle centered at the origin (0,0).
    """

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
    """Class to create VelocityRectangles that project agent's velocity by a factor to calculate social metric violations"""

    # k=3, how many time steps ahead to extend velocity rectangle by
    LENGTH_SCALE = 3
    WIDTH_SCALE = 1

    def __init__(self, agent: Agent = None):
        self._agent = agent
        assert agent is not None

        # in agent.py [self.px, self.py, self.vx, self.vy, self.radius]
        properties = self._agent.get_observable_state_list()
        vx, vy = properties[2:4]
        radius = properties[4]
        rwidth = 2 * radius * self.WIDTH_SCALE
        rlength = self.LENGTH_SCALE * (vx ** 2 + vy ** 2) ** 0.5

        # NOTE: DO NOT USE self._agent.theta as it is never updated
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
    """Class to create NormZoneRectangle (aka social norm zones) as suggested in SARL/LM-SARL paper"""

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
