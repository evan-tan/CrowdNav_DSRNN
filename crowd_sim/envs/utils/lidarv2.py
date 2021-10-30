# %%
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from crowd_sim.envs.utils.get_intersect import get_intersect
from crowd_sim.envs.utils.helper import create_agents_arr, unsqueeze, vec_norm


def rescale_angle(theta):
    """Rescale values from np.arctan2 output [np.pi,-np.pi] to [0,2*np.pi]"""
    return (theta + 2 * np.pi) % (2 * np.pi)


def get_valid_angles(sensor_pos: List[float], agent_xyr: np.ndarray) -> np.ndarray:
    """Get valid angles
    NOTE: this makes the LiDAR implementation much faster since it only checks relevant polygons

    :param sensor_pos: [x,y] sensor position in WORLD FRAME
    :type sensor_pos: List[float]
    :param agent_xyr: agent attributes, format: [x,y,radius]
    :type agent_xyr: np.ndarray
    :return: min/max angles for each agent obstacle for LiDAR \
        to hit obstacle shape=(n_agents, 2)
    :rtype: np.ndarray
    """

    # center RELATIVE TO SENSOR FRAME
    rel_x = agent_xyr[:, 0] - sensor_pos[0]
    rel_y = agent_xyr[:, 1] - sensor_pos[1]

    # scale to [0,2pi]
    heading = rescale_angle(np.arctan2(rel_y, rel_x))
    # determine min/max angles for lidar to hit obstacle
    dx = agent_xyr[:, 2] * np.sin(heading)
    dy = agent_xyr[:, 2] * np.cos(heading)
    ax = rel_x + dx
    ay = rel_y - dy
    bx = rel_x - dx
    by = rel_y + dy

    # unsorted array
    minmax_arr = np.zeros((2, agent_xyr.shape[0]))
    # NOTE: it is crucial to adjust angle here according to sensor heading for min/max angle values
    minmax_arr[0, :] = rescale_angle(np.arctan2(ay, ax))
    minmax_arr[1, :] = rescale_angle(np.arctan2(by, bx))

    # preserve order by changing (2,n_agents) -> (n_agents,2)
    # sort each row so min = index 0, max = index 1
    return np.sort(minmax_arr, axis=0).T


def get_valid_angle_idx(
    test_theta: float, test_range: np.array, use_radians: bool = True
):
    """Test a SINGLE angle is within specified min/max valid angles"""
    n_tests = test_range.shape[0]
    div = 2 * np.pi if use_radians else 360

    # preprocess, andle cases where obstacle occupies range at 0 deg
    lower = test_range[:, 0]
    upper = test_range[:, 1]
    # transform test theta to np array
    test_theta = np.ones(n_tests) * test_theta
    idx = np.where(upper - lower >= np.pi)
    upper[idx] -= 2 * np.pi
    test_theta[idx] -= 2 * np.pi
    # swap "lower" and "upper" to make sure lower < upper
    lower[idx], upper[idx] = upper[idx], lower[idx]

    res = (test_theta - lower) % div < (upper - lower) % div
    return res


def rotate(arr: np.ndarray, theta: float, ref_pt: Tuple[float] = [0, 0]) -> np.ndarray:
    """Perform 2D rotation about a reference point

    :param arr: array of xy points, shape=(2, n_pts)
    :type arr: np.ndarray
    :param theta: Rotation angle in radians
    :type theta: float
    :param ref_pt: xy-coordinate of reference frame, defaults to [0, 0]
    :type ref_pt: Tuple[float], optional
    :raises NotImplementedError: Not suitable for 3D tensors
    :return: Rotated line
    :rtype: np.ndarray
    """

    c, s = np.cos(theta), np.sin(theta)
    rot_arr = np.zeros_like(arr)
    if arr.ndim == 2:
        dx = arr[0, :] - ref_pt[0]
        dy = arr[1, :] - ref_pt[1]
        rot_arr[0, :] = c * dx - s * dy + ref_pt[0]
        rot_arr[1, :] = s * dx + c * dy + ref_pt[1]
        return rot_arr
    # preprocessing if vectorized
    elif arr.ndim == 3:
        raise NotImplementedError


def create_lidarbeam_arr(
    sensor_pos: List[float],
    sensor_heading: float,
    max_range: float,
    n_beams: int,
    n_pts: int = 600,
    resolution: float = None,
) -> Tuple[np.ndarray]:
    """Create all lidar beam points in the WORLD FRAME

    :param sensor_pos: Sensor xy position in world frame
    :type sensor_pos: List[float]
    :param sensor_heading: Sensor orientation in world frame
    :type sensor_heading: float
    :param max_range: Maximum sensor range
    :type max_range: float
    :param n_beams: Number of LiDAR beams
    :type n_beams: int
    :param n_pts: Number of points to represent LiDAR beam line, defaults to 600
    :type n_pts: int, optional
    :param resolution: Spacing between 2 points on LiDAR line, when specified will OVERRIDE n_pts param, defaults to None
    :type resolution: float, optional
    :return: LiDAR beam array shape=(n_beams, 2, n_pts per beam), and corresponding angles array
    :rtype: Tuple[np.ndarray]
    """

    # if set, this determines how accurate your measurements are
    if resolution is not None:
        # override n_pts to achieve xy_resolution
        n_pts = max_range / resolution
        n_pts = int(n_pts) + 1 if n_pts % 1 != 0 else int(n_pts)

    # points depend on resolution of line (lidar beam)
    single_beam = np.zeros((2, n_pts))
    single_beam[0, :] = np.linspace(0, max_range, n_pts)
    single_beam[1, :] = np.linspace(0, 0, n_pts)

    # (2,n_pts) -> (1,2,n_pts)
    single_beam = unsqueeze(single_beam, dim=0)
    # (n_beams, 2, n_pts)
    rot_beams = np.repeat(single_beam, n_beams, axis=0)
    lidar_angles = np.linspace(0, 2 * np.pi, n_beams)
    # rotate according to sensor
    lidar_angles = rescale_angle(lidar_angles + sensor_heading)

    # TODO: vectorize
    # rotate each beam to the correct angle
    for i in range(rot_beams.shape[0]):
        rot_beams[i] = rotate(rot_beams[i], lidar_angles[i], ref_pt=[0, 0])

    # translate x and y components
    rot_beams[:, 0, :] += sensor_pos[0]
    rot_beams[:, 1, :] += sensor_pos[1]
    return rot_beams, lidar_angles


def check_dist_collision_polygon(
    sensor_pos: List[float],
    beam_pts: np.ndarray,
    obst_xyr: np.ndarray,
) -> Tuple[float]:
    """Check if a SINGLE beam collides with selected obstacles, in the WORLD FRAME

    :param sensor_pos: LiDAR sensor xy position
    :type sensor_pos: List[float]
    :param beam_pts: LiDAR beams array (n_beams, 2, n_pts per beam)
    :type beam_pts: np.ndarray
    :param obst_xyr: Obstacle attributes (x,y,radius)
    :type obst_xyr: np.ndarray
    :return: xy point in world frame, where LiDAR has collided with object
    :rtype: Tuple[float]
    """

    center, radius = obst_xyr[:, 0:2], obst_xyr[:, 2]
    # determine closest point to sensor if multiple points
    sensor_pos = np.array(sensor_pos)
    rel_distances = np.linalg.norm(center - sensor_pos, axis=1)
    # select closest obstacle center
    center_idx = np.argmin(rel_distances)
    # select appropriate obstacle
    center = center[center_idx, :]
    radius = radius[center_idx]

    # traverse along lidar beam
    # NOTE: DO NOT translate with sensor pos since beam is already in world frame

    dx = (center[0] - beam_pts[0, :]) ** 2
    dy = (center[1] - beam_pts[1, :]) ** 2
    res = np.sqrt(dx + dy)

    # values that are all valid collisions, returns a (1,tuple(..))
    val_idx = np.where(res < radius)

    if len(val_idx[0]) > 0:
        # grab closest point to sensor, this handles occlusion
        end_idx = val_idx[0].min()
        end_pt = beam_pts[:, end_idx].squeeze()
        return end_pt
    else:
        return None


def process_obstacles(
    cfg: dict,
    sensor_pos: List[float],
    beam_arr: np.ndarray,
    obst_xyr: np.ndarray,
    lidar_angles: np.ndarray,
    valid_angles: np.ndarray,
    wall_pts: List[Tuple],
) -> np.ndarray:
    """Determine end points for each LiDAR beam

    :param cfg: Dict containing sensor attributes
    :type cfg: dict
    :param sensor_pos: LiDAR sensor xy position
    :type sensor_pos: List[float]
    :param beam_arr: All LiDAR beams, shape=(n_beams,2, n_pts per beam)
    :type beam_arr: np.ndarray
    :param obst_xyr: All corresponding obstacle attributes for obst_arr, where first dimension is aligned
    :type obst_xyr: np.ndarray
    :param lidar_angles: All LiDAR angles
    :type lidar_angles: np.ndarray
    :param valid_angles: Output from get_valid_angles()
    :type valid_angles: np.ndarray
    :param wall_pts: List of xy points defining walls
    :type wall_pts: List[Tuple]
    :return: xy position where each beam ends in world frame, shape=(n_beams,2)
    :rtype: np.ndarray
    """
    assert "max_range" in cfg.keys()
    lidar_end_pts = np.zeros((beam_arr.shape[0], 2))

    # TODO: vectorize
    for i in range(beam_arr.shape[0]):
        beam_xy = beam_arr[i, :, :]
        beam_start = beam_arr[i, :, 0]  # at sensor pos
        beam_end = beam_arr[i, :, -1]  # @ max sensor range

        end_pt = beam_end

        # ensure wrap around
        if wall_pts[-1] != wall_pts[0]:
            wall_pts.append(wall_pts[0])

        valid_angle_idx = get_valid_angle_idx(lidar_angles[i], valid_angles)
        # passed angle check, but might still fail distance checks
        alt_tmp_pt = None
        if valid_angle_idx.astype(np.int).sum() > 0:
            valid_obst_xyr = obst_xyr[valid_angle_idx]
            alt_tmp_pt = check_dist_collision_polygon(
                sensor_pos, beam_xy, valid_obst_xyr
            )

        # if failed distance check, return val = None
        if alt_tmp_pt is not None:
            end_pt = alt_tmp_pt
        else:
            # check for collisions with wall
            tmp_pt = None
            for j in range(len(wall_pts) - 1):
                # check for intersection with walls
                intersect_pt = get_intersect(
                    beam_start, beam_end, wall_pts[j], wall_pts[j + 1]
                )
                # each lidar beam only intersects with single wall (single line)
                if intersect_pt is not None:
                    tmp_pt = intersect_pt
                    break

            if tmp_pt is not None:
                if vec_norm(tmp_pt, sensor_pos) <= cfg["max_range"]:
                    end_pt = tmp_pt

        # lidar beams cannot end inside robot's radius
        robot_r = cfg["robot_radius"]
        if vec_norm(end_pt, sensor_pos) < robot_r:
            x = sensor_pos[0] + robot_r * np.cos(lidar_angles[i])
            y = sensor_pos[1] + robot_r * np.sin(lidar_angles[i])
            end_pt = np.array([x, y])

        # store result
        lidar_end_pts[i] = end_pt

    return lidar_end_pts


class LidarSensor:
    """LiDAR Sensor purely using NumPy, about 4x faster than Shapely
    NOTE: distances are min/max scaled according to sensor range!!!
    How to use
    - parse cfg into constructor
    - parse agent obstacles and wall obstacles in CrowdSim / CrowdSimDict reset()
    - call update_sensor() in CrowdSim / CrowdSimDict step()
    - call lidar_spin() to get points
    """

    N_PTS_POLY = 100
    BEAM_RESOLUTION = 0.01

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sensor_pos = [0, 0]  # sensor location
        self.sensor_heading = 0.0  # sensor heading
        self.MAX_DIST = cfg["max_range"]

        # dynamic obstacles
        self.agent_arr = None  # np array containing all agent polygon points
        self.agent_attrs = None  # (x,y,radius) of each agent
        self.valid_angles = None  # valid min/max angles of each dynamic obstacle in order for lidar to hit the object

        # static obstacles
        self.wall_pts = None  # (n_vertices, 2) of x,y points
        self.viz = {}  # dict to store info

    def parse_obstacles(self, obst_pts, mode=""):
        """If mode == 'agents', type(obst_pts) should be np.ndarray, otherwise if mode == 'walls', type(obst_pts) should be list"""
        if mode == "agents":
            if type(obst_pts) == list:
                obst_pts = np.array(obst_pts)
            assert type(obst_pts) is np.ndarray
            assert obst_pts.shape[1] == 3
            self.agent_arr = create_agents_arr(obst_pts, n_pts=self.N_PTS_POLY)
            self.agent_attrs = obst_pts

        elif mode == "walls":
            assert type(obst_pts) == list
            # ensure wrap around
            if obst_pts[-1] != obst_pts[0]:
                obst_pts.append(obst_pts[0])
            self.wall_pts = obst_pts

    def update_sensor(self, xy=None, heading=None):
        if xy is not None:
            self.sensor_pos = xy
        if heading is not None:
            self.sensor_heading = heading

    def sensor_spin(self, normalize=True) -> Tuple[np.ndarray]:
        beams_arr, rot_lidar_angles = create_lidarbeam_arr(
            self.sensor_pos,
            self.sensor_heading,
            self.cfg["max_range"],
            self.cfg["num_beams"],
            resolution=0.01,
        )
        valid_angles = get_valid_angles(self.sensor_pos, self.agent_attrs)
        lidar_end_pts = process_obstacles(
            self.cfg,
            self.sensor_pos,
            beams_arr,
            self.agent_attrs,
            rot_lidar_angles,
            valid_angles,
            self.wall_pts,
        )
        # determine angles and distances relative to sensor position
        rel_end_pts = lidar_end_pts - self.sensor_pos
        rel_dist = np.linalg.norm(rel_end_pts, axis=1)

        # scale to 0-1 based on max sensor range
        if normalize:
            rel_dist = np.clip(rel_dist / self.cfg["max_range"], 0, 1)

        # adjust back to world frame
        Fw_lidar_angles = rot_lidar_angles - self.sensor_heading

        return Fw_lidar_angles, rel_dist, lidar_end_pts


if __name__ == "__main__":
    # format: (x,y, radius)

    t = 20 / 2
    wall_pts = [(-t, -t), (t, -t), (t, t), (-t, t)]
    robot_radius = 0.3
    cfg = {"max_range": 11, "num_beams": 180, "robot_radius": robot_radius}

    sensor = LidarSensor(cfg)

    # DEFINE
    sensor_pos = (1, 1)
    sensor_heading = np.pi / 4
    n_agents = 5

    n_iter = 1e2
    start = time.time()

    for i in range(int(n_iter)):
        agent_xyr = [
            (2, 2, 0.5),
            (-4, -3, 0.5),
            (-3, -3, 0.3),
            (-3, -4, 0.2),
            (-0.5, 0.5, 0.4),
            (5, -5, 0.3),
        ]
        agent_xyr = np.array(agent_xyr)
        agent_arr = create_agents_arr(agent_xyr, n_pts=100)
        beam_arr, rot_angles = create_lidarbeam_arr(
            sensor_pos,
            sensor_heading,
            cfg["max_range"],
            cfg["num_beams"],
            resolution=0.01,
        )
        valid_theta = get_valid_angles(sensor_pos, agent_xyr)
        retval = process_obstacles(
            cfg,
            sensor_pos,
            beam_arr,
            agent_xyr,
            rot_angles,
            valid_theta,
            wall_pts,
        )
        rel_end_pts = retval - sensor_pos
        rel_dist = np.linalg.norm(rel_end_pts, axis=1)
        rel_dist = (rel_dist - rel_dist.min()) / (cfg["max_range"] - rel_dist.min())
    elapsed = time.time() - start
    print(f"{n_iter} iterations took {elapsed:.6f} seconds")

    # DEBUGGING
    fig, ax = plt.subplots(figsize=(7, 7))
    val = 20 / 2 + 5
    ax.set_xlim(-val, val)
    ax.set_ylim(-val, val)
    ax.set_xlabel("x(m)", fontsize=16)
    ax.set_ylabel("y(m)", fontsize=16)

    # plot lidar beams
    x_pts = beam_arr[:, 0, :].flatten()
    y_pts = beam_arr[:, 1, :].flatten()
    ax.plot(x_pts, y_pts, color="r", alpha=0.1)

    # plot end points
    end_x, end_y = retval[:, 0], retval[:, 1]
    ax.scatter(end_x, end_y, marker="h", color="k", s=0.5)
    ax.scatter(end_x[0], end_y[0], marker="h", color="r", s=10)
    sensor = plt.Circle(sensor_pos, radius=0.25, color="g")
    ax.add_patch(sensor)

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i in range(agent_arr.shape[0]):
        ax.scatter(
            agent_arr[i, 0, :].flatten(),
            agent_arr[i, 1, :].flatten(),
            color=colors[i],
            s=0.01,
        )

        # # show spatial data for each obstacle
        tmp_line = np.zeros((2, 30))
        dist = np.linalg.norm(agent_xyr[i, :2] - sensor_pos)
        tmp_line[0, :] = np.linspace(0, dist, 30)
        tmp_line[1, :] = np.linspace(0, 0, 30)
        min_line = rotate(tmp_line, valid_theta[i, 0])
        max_line = rotate(tmp_line, valid_theta[i, 1])
        min_line += unsqueeze(sensor_pos, dim=1)
        max_line += unsqueeze(sensor_pos, dim=1)

        ax.scatter(min_line[0, :], min_line[1, :], alpha=0.5, color=colors[i], s=0.1)
        ax.scatter(max_line[0, :], max_line[1, :], color=colors[i], s=0.1)

    range_str = f"Max Sensor Range = {int(cfg['max_range'])}m"
    ax.annotate(
        range_str,
        xy=(210, 25),
        xycoords="axes points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", fc="w"),
    )
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Lidar Angles (degrees)")
    ax1.set_ylabel("Normalized Distance")
    ax1.plot(np.rad2deg(rescale_angle(rot_angles - sensor_heading)), rel_dist)
    ax1.annotate(
        range_str,
        xy=(210, 75),
        xycoords="axes points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", fc="w"),
    )

    plt.show()

# %%
