#%%
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from crowd_sim.envs.utils.get_intersect import get_intersect
from crowd_sim.envs.utils.helper import vec_norm

# import numba as nb
# from numba import extending, jit, njit

# TODO: make OOP
def rescale_angle(theta):
    """Rescale values from np.arctan2 output [np.pi,-np.pi] to [0,2*np.pi]"""
    return (theta + 2 * np.pi) % (2 * np.pi)


def get_valid_angles(sensor_pos, sensor_heading, agent_xyr: np.ndarray):
    """Get valid angles relative to SENSOR FRAME"""
    # center RELATIVE TO SENSOR FRAME
    cx = agent_xyr[:, 0] - sensor_pos[0]
    cy = agent_xyr[:, 1] - sensor_pos[1]

    # scale to [0,2pi]
    heading = rescale_angle(np.arctan2(cy, cx))
    # determine min/max angles for lidar to hit obstacle
    dx = agent_xyr[:, 2] * np.sin(heading)
    dy = agent_xyr[:, 2] * np.cos(heading)
    ax = cx + dx
    ay = cy - dy
    bx = cx - dx
    by = cy + dy
    minmax_arr = np.empty((2, agent_xyr.shape[0]))
    # NOTE: it is crucial to adjust angle here for min/max angle values
    minmax_arr[0, :] = rescale_angle(np.arctan2(ay, ax) - sensor_heading)
    minmax_arr[1, :] = rescale_angle(np.arctan2(by, bx) - sensor_heading)

    # preserve order by changing (2,n_agents) -> (n_agents,2)
    # sort each row so min = index 0, max = index 1
    return np.sort(minmax_arr, axis=0).T


def rotate(arr: np.ndarray, theta: float):
    """Perform 2D rotation about the ORIGIN
    :param array: array of points, each point has format (x,y)
    :type array: np.ndarray, shape=(2, n_pts)
    :param theta: Rotation angle (radians)
    :type theta: float
    :return: Rotated line
    :rtype: np.ndarray
    """
    # rot matrix, shape = (2,2)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]]).astype(np.float64)
    # do matrix multiplication, inner dims have to match
    assert rot.shape[1] == arr.shape[0]
    return rot @ arr


def unsqueeze(array, dim):
    """Wrapper function for torch.unsqueeze() functionality in NumPy"""
    return np.expand_dims(array, axis=dim)


def create_agents_arr(agent_pts: np.ndarray, n_pts):
    """Create all agent points in the WORLD FRAME
    :param agent_pts: All points for agent obstacles
    :type agent_pts: np.ndarray, shape=(n_agents, 3)
    :param n_pts: number of points to represent agent obstacle polygon
    :type n_pts: int
    :return: Array of xy points for each agent polygon
    :rtype: np.ndarray, shape=(n_agent,2, n_pts)
    """
    # points to represent polygon
    poly_angles = np.linspace(0, 2 * np.pi, n_pts)
    # (n_agents,) -> (n_agents,1)
    radii = unsqueeze(agent_pts[:, 2], dim=1)
    px = unsqueeze(agent_pts[:, 0], dim=1)
    py = unsqueeze(agent_pts[:, 1], dim=1)

    # (n_agents,2,n_pts)
    agent_arr = np.empty((agent_pts.shape[0], 2, n_pts))
    # broadcasting, position in WORLD FRAME
    agent_arr[:, 0] = px + radii * np.cos(poly_angles)
    agent_arr[:, 1] = py + radii * np.sin(poly_angles)
    return agent_arr


def create_lidarbeam_arr(
    sensor_pos, sensor_heading, max_range, n_spacings, n_pts=600, resolution=None
):
    """Create all lidar beam points in the WORLD FRAME

    :param sensor_pos: xy position
    :type sensor_pos: List[float]
    :param max_range: max sensor range
    :type max_range: float
    :param n_spacings: number of LiDAR beams
    :type n_spacings: int
    :param n_pts: Number of points to represent LiDAR beam line, defaults to 600
    :type n_pts: int, optional
    :param resolution: Spacing between 2 points on LiDAR line, defaults to None
    :type resolution: float, optional
    :return: LiDAR beam array and corresponding angles
    :rtype: np.ndarray, shape=(n_beams, 2, n_pts per beam)
    """
    if resolution is not None:
        # override n_pts to achieve resolution
        n_pts = int(max_range / resolution) + 1

    # points depend on resolution of line (lidar beam)
    single_beam = np.zeros((2, n_pts))
    # translate according to sensor heading
    # single_beam[0, :] = sensor_pos[0] + np.linspace(0, cfg["max_range"], n_pts)
    single_beam[0, :] = np.linspace(0, max_range, n_pts)
    single_beam[1, :] = np.linspace(0, 0, n_pts)

    # (2,n_pts) -> (1,2,n_pts)
    single_beam = unsqueeze(single_beam, dim=0)
    # (n_beams, 2, n_pts)
    all_beams = np.repeat(single_beam, n_spacings, axis=0)
    lidar_angles = np.linspace(0, 2 * np.pi, n_spacings)
    # all_beams = np.repeat(single_beam, cfg["num_spacings"], axis=0)
    # lidar_angles = np.linspace(0, 2 * np.pi, cfg["num_spacings"])

    # # TODO: vectorize
    # # (n_beams, 2, n_pts)
    rot_beams = np.zeros((lidar_angles.shape[0], 2, n_pts))
    for i in range(rot_beams.shape[0]):
        rot_beams[i] = rotate(all_beams[i], lidar_angles[i] + sensor_heading)

    # translate x and y components
    rot_beams[:, 0, :] += sensor_pos[0]
    rot_beams[:, 1, :] += sensor_pos[1]
    return rot_beams, lidar_angles


def check_collision_polygon(
    sensor_pos, beam_pts: np.ndarray, obst_pts: np.ndarray, obst_attrs: np.ndarray
):
    """Check if a SINGLE beam collides with selected obstacles, in the world frame

    :param sensor_pos: [description]
    :type sensor_pos: [type]
    :param beam_pts: (2,N) points of LiDAR beam
    :type beam_pts: np.ndarray
    :param obst_pts: (2,M) points for selected obstacles, where M = n_obst * n_pts per obstacle
    :type obst_pts: np.ndarray
    :param obst_attrs: (num_obstacles, 3), [x,y,radius] for each obst
    :type obst_attrs: np.ndarray
    :return: [description]
    :rtype: [type]
    """

    center, radius = obst_attrs[:, 0:2], obst_attrs[:, 2]
    # determine closest point if multiple points
    rel_distances = np.linalg.norm(center - sensor_pos, axis=1)
    # select closer obstacle center
    center_idx = np.argmin(rel_distances)

    # select appropriate obstacle
    center = center[center_idx, :]
    radius = radius[center_idx]
    # sel_obst = obst_pts[center_idx]
    # sel_attrs = obst_attrs[center_idx]

    end_pt = np.empty((2, 1))
    # traverse along lidar beam, DO NOT translate with sensor pos since beam is already in world frame
    dx = (center[0] - beam_pts[0, :]) ** 2
    dy = (center[1] - beam_pts[1, :]) ** 2

    res = np.sqrt(dx + dy)
    # idx = np.searchsorted(res, radius)
    # values that are all valid collisions, returns a (1,tuple(..))
    val_idx = np.where(res <= radius)
    if len(val_idx[0]) > 0:
        end_idx = val_idx[0].min()
        end_pt = beam_pts[:, end_idx]

    return end_pt


def process_obstacles(
    cfg, sensor_pos, beam_arr, obst_arr, obst_xyr, lidar_angles, valid_angles, wall_pts
):
    lidar_end_pts = np.empty(beam_arr.shape[0:2])

    # TODO: vectorize
    # loop through all beams
    for i in range(beam_arr.shape[0]):
        # check that lidar angles are within valid max/min angles before checking collision points
        # index (0,1) correspond to (min,max) valid angles
        # valid angles are in sensor frame, but since SENSOR FRAME and WORLD FRAME are aligned we can use them directly
        min_cond = lidar_angles[i] >= valid_angles[:, 0]
        max_cond = lidar_angles[i] <= valid_angles[:, 1]
        valid_indices = np.bitwise_and(min_cond, max_cond)
        # for each valid obstacle check for collision pt
        valid_obst = obst_arr[valid_indices]
        valid_obst_attr = obst_xyr[valid_indices]

        beam_xy = beam_arr[i, :, :]
        beam_ptA = beam_arr[i, :, 0]  # at sensor pos
        beam_ptB = beam_arr[i, :, -1]  # @ max sensor range
        end_pt = beam_ptB  # by default set to max range of lidar
        alt_end_pt = None
        # if collisions among DYNAMIC OBSTACLES detected
        if valid_obst.size > 0:
            # (n_obst, 2, n_pts)
            # determine the end point based on the beam
            # dist_check = np.vstack((valid_obst.min(), valid_obst.max()))
            # dist_check = np.linalg.norm(dist_check, axis=1)
            # dynamic_available = (dist_check <= cfg["max_range"]).any()
            end_pt = check_collision_polygon(
                sensor_pos, beam_xy, valid_obst, valid_obst_attr
            ).squeeze()
        # check for collisions with wall
        # ensure wrap around
        if wall_pts[-1] != wall_pts[0]:
            wall_pts.append(wall_pts[0])

        for j in range(len(wall_pts) - 1):
            # check for intersection with walls
            intersect_pt = get_intersect(
                beam_ptA, beam_ptB, wall_pts[j], wall_pts[j + 1]
            )
            # each lidar beam only intersects with single wall
            if intersect_pt is not None:
                alt_end_pt = intersect_pt
                break
        if vec_norm(end_pt, sensor_pos) > cfg["max_range"]:
            end_pt = beam_ptB
        if alt_end_pt is not None:
            if vec_norm(alt_end_pt, sensor_pos) < vec_norm(end_pt, sensor_pos):
                end_pt = alt_end_pt

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
        self.viz = {"valid_angles": None, "global_end_pts": None}

    def parse_obstacles(self, obst_pts, mode=""):
        if mode == "agents":
            if type(obst_pts) == list:
                obst_pts = np.array(obst_pts)
            self.agent_arr = create_agents_arr(obst_pts, n_pts=45)
            self.agent_attrs = obst_pts

        elif mode == "walls":
            # ensure wrap around
            if obst_pts[-1] != obst_pts[0]:
                obst_pts.append(obst_pts[0])
            self.wall_pts = obst_pts

    # TODO: update existing arrays based on d_xy, d_heading
    def update_sensor(self, xy=None, heading=None):
        if xy is not None:
            self.sensor_pos = xy

        if heading is not None:
            self.sensor_heading = heading

    def sensor_spin(self, normalize=True) -> np.ndarray:
        beams_arr, lidar_angles = create_lidarbeam_arr(
            self.sensor_pos,
            self.sensor_heading,
            self.cfg["max_range"],
            self.cfg["num_spacings"],
            n_pts=600,
        )
        valid_angles = get_valid_angles(
            self.sensor_pos, self.sensor_heading, self.agent_attrs
        )
        self.viz["valid_angles"] = valid_angles

        # end points in WORLD FRAME
        lidar_end_pts = process_obstacles(
            self.cfg,
            self.sensor_pos,
            beams_arr,
            self.agent_arr,
            self.agent_attrs,
            lidar_angles,
            valid_angles,
            self.wall_pts,
        )
        # determine angles and distances relative to sensor position
        rel_end_pts = lidar_end_pts - self.sensor_pos
        rel_dist = np.linalg.norm(rel_end_pts, axis=1)
        self.viz["global_end_pts"] = lidar_end_pts
        self.viz["relative_end_pts"] = rel_end_pts

        # minmax scale to 0-1 based on max sensor range
        if normalize:
            rel_dist = (rel_dist - rel_dist.min()) / (
                self.cfg["max_range"] - rel_dist.min()
            )

        return lidar_angles, rel_dist

    def get_viz_pts(self, key):
        if key not in self.viz.keys():
            return None
        else:
            if key == "global_end_pts":
                return self.viz["global_end_pts"]


if __name__ == "__main__":
    # format: (x,y, radius)
    agent_xyr = [
        (1, 1, 0.5),
        (2, 2, 0.5),
        (-4, -3, 0.5),
        (-3, -3, 0.3),
        (-3, -4, 0.2),
        (-0.5, 0.5, 0.4),
        (5, -5, 0.3),
    ]
    agent_xyr = np.array(agent_xyr)
    cfg = {"max_range": 11, "num_spacings": 180}
    n_spacings = cfg["num_spacings"]
    max_range = cfg["max_range"]
    t = 20 / 2
    wall_pts = [(-t, -t), (t, -t), (t, t), (-t, t)]
    n_iter = 1e2
    start = time.time()
    sensor_pos = [0, 0]
    sensor_heading = 0

    for i in range(int(n_iter)):
        agents = create_agents_arr(agent_xyr, n_pts=100)
        beams, angles = create_lidarbeam_arr(
            sensor_pos, sensor_heading, max_range, n_spacings, n_pts=300
        )
        valid_theta = get_valid_angles(sensor_pos, sensor_heading, agent_xyr)
        retval = process_obstacles(
            cfg, sensor_pos, beams, agents, agent_xyr, angles, valid_theta, wall_pts
        )
        rel_end_pts = retval - sensor_pos
        rel_dist = np.linalg.norm(rel_end_pts, axis=1)
        rel_dist = (rel_dist - rel_dist.min()) / (cfg["max_range"] - rel_dist.min())

    elapsed = time.time() - start
    print(f"{n_iter} iterations took {elapsed:.6f} seconds")

    # # DEBUGGING
    fig, ax = plt.subplots(figsize=(7, 7))
    val = 20 / 2 + 5
    ax.set_xlim(-val, val)
    ax.set_ylim(-val, val)
    ax.set_xlabel("x(m)", fontsize=16)
    ax.set_ylabel("y(m)", fontsize=16)

    # plot lidar beams
    x_pts = beams[:, 0, :].flatten()
    y_pts = beams[:, 1, :].flatten()
    ax.plot(x_pts, y_pts, color="r", alpha=0.1)

    # # plot end points
    end_x = retval[:, 0]
    end_y = retval[:, 1]

    ax.scatter(end_x, end_y, marker="h", color="k", s=0.5)
    ax.scatter(end_x[0], end_y[0], marker="h", color="r", s=6)

    sensor = plt.Circle(sensor_pos, radius=0.3, color="g")
    ax.add_patch(sensor)
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i in range(agents.shape[0]):
        # plot agent polygons
        agent_x = agents[i, 0, :].flatten()
        agent_y = agents[i, 1, :].flatten()
        ax.scatter(agent_x, agent_y, color=colors[i], s=0.01)

    fig1, ax1 = plt.subplots()
    ax1.plot(np.rad2deg(angles), rel_dist)
    plt.show()
    #     # show spatial data for each obstacle
    #     tmp_line = np.empty((2, 30))
    #     dist = np.linalg.norm(agent_xyr[i, :2])
    #     tmp_line[0, :] = np.linspace(0, dist, 30)
    #     tmp_line[1, :] = np.linspace(0, 0, 30)
    #     min_line = rotate(tmp_line, valid_theta[i, 0])
    #     max_line = rotate(tmp_line, valid_theta[i, 1])
    #     min_line += unsqueeze(sensor_pos, dim=1)
    #     max_line += unsqueeze(sensor_pos, dim=1)
    #     ax.scatter(min_line[0, :], min_line[1, :], alpha=0.5, color=colors[i], s=0.1)
    #     ax.scatter(max_line[0, :], max_line[1, :], color=colors[i], s=0.1)
    # # for j in range(beams.shape[0]):
    # #     # annotate lidar
    # #     ax.text(end_x[j], end_y[j], str(j))

    # plt.show()

# %%
