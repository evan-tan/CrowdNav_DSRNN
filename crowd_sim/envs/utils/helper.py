from typing import List

import matplotlib.lines as mlines
import numpy as np
import shapely.geometry

# from crowd_sim.envs.utils.agent import Agent
from shapely.geometry import LineString


def unsqueeze(arr: np.ndarray, dim: int) -> np.ndarray:
    """Wrapper function for torch.unsqueeze() functionality in NumPy"""
    return np.expand_dims(arr, axis=dim)


def wrap_angle(angle):
    """Wrap angle between [-180, 180]"""
    while angle <= -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle


def vec_norm(A, B):
    """Small wrapper to calculate L2 norm between 2 points

    :param pt1: point in [x,y] format
    :type pt1: list
    :param pt2: point in [x,y] format
    :type pt2: list
    :return: norm2 of points
    :rtype: np.float64
    """
    return np.linalg.norm([A[0] - B[0], A[1] - B[1]])


def ang_diff(pt1, pt2):
    """Calculate arctan2 between 2 points

    :param pt1: point in [x,y] format
    :type pt1: list
    :param pt2: point in [x,y] format
    :type pt2: list
    :return: arctan2 of points
    :rtype: np.float
    """
    return np.arctan2([pt1[1] - pt2[1], pt1[0] - pt2[0]])


def check_inside_world(agent_ellipse, wall_pts):
    """Very jank and only used to check if robot is inside world"""
    if wall_pts[-1] != wall_pts[0]:
        wall_pts.append(wall_pts[0])
    # decompose wall boundaries i.e. polygon, into line strings
    is_inside = True
    for i in range(len(wall_pts) - 1):
        line = LineString([wall_pts[i], wall_pts[i + 1]])
        is_intersecting = line.intersection(agent_ellipse)
        if not is_intersecting.is_empty:
            is_inside = False
            break

    return is_inside


############################# event logging #############################
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


############################# point generation #############################
def rand_world_pt(config):
    """Generate random world point based on square_width"""
    # this assumes world is centered at (0,0)
    return (np.random.random() - 0.5) * config.sim.square_width / 2


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
