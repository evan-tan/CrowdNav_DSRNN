import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from crowd_sim.envs.utils.get_intersect import get_intersect

# import numba as nb
# from numba import extending, jit, njit


def rescale_angle(theta):
    return (theta + 2 * np.pi) % (2 * np.pi)


def valid_angles(agent_xyr: np.ndarray):
    # center
    cx, cy = agent_xyr[:, 0], agent_xyr[:, 1]
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
    minmax_arr[0, :] = rescale_angle(np.arctan2(ay, ax))
    minmax_arr[1, :] = rescale_angle(np.arctan2(by, bx))

    # preserve order by changing (2,n_agents) -> (n_agents,2)
    # sort each row so min = index 0, max = index 1
    return np.sort(minmax_arr, axis=0).T


def rotate(arr: np.ndarray, theta: float):
    """Perform 2D rotation
    :param array: array of points, each point has format (x,y)
    :type array: np.ndarray (2, n_points)
    :param theta: Rotation angle (radians)
    :type theta: float
    :return: Rotated line
    :rtype: [type]
    """
    # rot matrix, shape = (2,2)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]]).astype(np.float64)

    # do matrix multiplication, inner dims have to match
    assert rot.shape[1] == arr.shape[0]

    # convert 2D to 3D
    return rot @ arr


def unsqueeze(array, dim):
    return np.expand_dims(array, axis=dim)


def create_agents_arr(agent_pts: np.ndarray, n_pts):
    # points to represent polygon
    poly_angles = np.linspace(0, 2 * np.pi, n_pts)
    # (7,) -> (7,1)
    radii = unsqueeze(agent_pts[:, 2], dim=1)
    px = unsqueeze(agent_pts[:, 0], dim=1)
    py = unsqueeze(agent_pts[:, 1], dim=1)

    # (7,2,n_pts)
    agent_arr = np.empty((agent_pts.shape[0], 2, n_pts))
    # broadcasting
    agent_arr[:, 0] = px + radii * np.cos(poly_angles)
    agent_arr[:, 1] = py + radii * np.sin(poly_angles)
    return agent_arr


def create_lidarbeam_arr(max_range, n_spacings, n_pts):
    # points depend on resolution of line (lidar beam)
    single_beam = np.zeros((2, n_pts))
    # translate according to sensor heading
    sensor_pos = (0, 0)
    # single_beam[0, :] = sensor_pos[0] + np.linspace(0, cfg["max_range"], n_pts)
    single_beam[0, :] = sensor_pos[0] + np.linspace(0, max_range, n_pts)
    single_beam[1, :] = sensor_pos[1] + np.linspace(0, 0, n_pts)

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
        rot_beams[i] = rotate(all_beams[i], lidar_angles[i])
    return rot_beams, lidar_angles


def check_collision_polygon(
    beam_pts: np.ndarray, obst_pts: np.ndarray, obst_attrs: np.ndarray
):
    """Check if a SINGLE beam collides with selected obstacles

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
    distances = np.linalg.norm(center, axis=1)
    # select closer obstacle center
    center_idx = np.argmin(distances)

    # select appropriate obstacle
    center = center[center_idx, :]
    radius = radius[center_idx]
    # sel_obst = obst_pts[center_idx]
    # sel_attrs = obst_attrs[center_idx]

    end_pt = np.empty((2, 1))
    # traverse along lidar beam
    dx = np.square(center[0] - beam_pts[0, :])
    dy = np.square(center[1] - beam_pts[1, :])

    res = np.sqrt(dx + dy)
    # idx = np.searchsorted(res, radius)
    # values that are all valid collisions, returns a (1,tuple(..))
    val_idx = np.where(res <= radius)
    if len(val_idx[0]) > 0:
        end_idx = val_idx[0].min()
        end_pt = beam_pts[:, end_idx]

    return end_pt


def process_obstacles(
    beam_arr, obst_arr, obst_xyr, lidar_angles, valid_angles, wall_pts
):
    lidar_end_pts = np.empty(beam_arr.shape[0:2])

    # TODO: vectorize
    # loop through all beams
    for i in range(beam_arr.shape[0]):
        # check that lidar angles are within valid max/min angles before checking collision points
        # index (0,1) correspond to (min,max) valid angles
        min_cond = lidar_angles[i] >= valid_angles[:, 0]
        max_cond = lidar_angles[i] <= valid_angles[:, 1]
        valid_indices = np.bitwise_and(min_cond, max_cond)
        # for each valid obstacle check for collision pt
        valid_obst = obst_arr[valid_indices]
        valid_obst_attr = obst_xyr[valid_indices]

        beam_xy = beam_arr[i, :, :]
        beam_ptA = beam_arr[i, :, 0]
        beam_ptB = beam_arr[i, :, -1]
        end_pt = beam_ptB
        # if collisions among DYNAMIC OBSTACLES detected
        if valid_obst.size > 0:
            # (n_obst, 2, n_pts)
            # determine the end point based on the beam
            # dist_check = np.vstack((valid_obst.min(), valid_obst.max()))
            # dist_check = np.linalg.norm(dist_check, axis=1)
            # dynamic_available = (dist_check <= cfg["max_range"]).any()
            end_pt = check_collision_polygon(
                beam_xy, valid_obst, valid_obst_attr
            ).squeeze()
        else:
            # check for collisions with wall
            # ensure wrap around
            if wall_pts[-1] != wall_pts[0]:
                wall_pts.append(wall_pts[0])

            for j in range(len(wall_pts) - 1):
                # check for intersection
                intersect_pt = get_intersect(
                    beam_ptA, beam_ptB, wall_pts[j], wall_pts[j + 1]
                )
                # each lidar beam only intersects with single wall
                if intersect_pt is not None:
                    end_pt = intersect_pt
                    break

        if np.linalg.norm(end_pt) >= cfg["max_range"]:
            end_pt = beam_ptB

        # store result
        lidar_end_pts[i] = end_pt
    return lidar_end_pts


if __name__ == "__main__":
    # format: (x,y, radius)
    agent_xyr = [
        (3.5, 3.5, 0.3),
        # (2, 2, 0.5),
        # (3, 2, 0.2),
        (-4, -3, 0.5),
        (-3, -3, 0.3),
        (-3, -4, 0.2),
        (4, -4, 0.3),
    ]
    agent_xyr = np.array(agent_xyr)
    cfg = {"max_range": 7, "num_spacings": 180}
    n_spacings = cfg["num_spacings"]
    max_range = cfg["max_range"]
    t = 20 / 4
    wall_pts = [(-t, -t), (t, -t), (t, t), (-t, t)]
    n_iter = 1e2
    start = time.time()
    for i in range(int(n_iter)):
        agents = create_agents_arr(agent_xyr, n_pts=45)
        beams, angles = create_lidarbeam_arr(max_range, n_spacings, n_pts=300)
        valid_theta = valid_angles(agent_xyr)
        retval = process_obstacles(
            beams, agents, agent_xyr, angles, valid_theta, wall_pts
        )

    elapsed = time.time() - start
    print(f"{n_iter} iterations took {elapsed:.6f} seconds")

    # DEBUGGING
    fig, ax = plt.subplots(figsize=(7, 7))
    val = 20 / 2 + 5
    ax.set_xlim(-val, val)
    ax.set_ylim(-val, val)
    ax.set_xlabel("x(m)", fontsize=16)
    ax.set_ylabel("y(m)", fontsize=16)

    x_pts = beams[:, 0, :].flatten()
    y_pts = beams[:, 1, :].flatten()

    # ax.plot(x_pts, y_pts, color="r")

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i in range(agents.shape[0]):
        agent_x = agents[i, 0, :].flatten()
        agent_y = agents[i, 1, :].flatten()
        ax.scatter(agent_x, agent_y, color=colors[i], s=0.01)

        # show spatial data for each obstacle
        line = np.empty((2, 30))
        line[0, :] = np.linspace(0, cfg["max_range"], 30)
        line[1, :] = np.linspace(0, 0, 30)
        # min_line = rotate(line, valid_theta[i, 0])
        # max_line = rotate(line, valid_theta[i, 1])
        # ax.scatter(min_line[0, :], min_line[1, :], alpha=0.5, color=colors[i], s=0.1)
        # ax.scatter(max_line[0, :], max_line[1, :], color=colors[i], s=0.1)

    end_x = retval[:, 0]
    end_y = retval[:, 1]
    ax.scatter(end_x, end_y, marker="x")
    np.savetxt("xy_coords.txt", retval)

    plt.show()
