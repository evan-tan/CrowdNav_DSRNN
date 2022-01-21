# %%
import random

import numpy as np
from crowd_sim.envs.utils.helper import (
    ang_diff,
    make_shapely_ellipse,
    rand_world_pt,
    vec_norm,
)
from shapely.geometry import Point, Polygon


def generate_offset(proposed_pos, size):
    """Generate offsets based on proposed position, used for checking if points are within a polygon

    :param proposed_pos: point in [x,y] format
    :type proposed_pos: list
    :param size: radius/length of obstacle/agent
    :type size: float
    :return: offset in [x,y]
    :rtype: list
    """
    # generate x offset
    if proposed_pos[0] < 0:
        x_offset = size
    else:
        x_offset = -size
    # generate y offset
    if proposed_pos[1] < 0:
        y_offset = size
    else:
        y_offset = -size

    return x_offset, y_offset


# check if a particular "test" geometry is inside "container"'s geometry
# True, if inside, False otherwise
def inside(test, container):
    if type(container) is list:
        # convert list of points into a polygon
        container = Polygon(container)

    if type(container) is Polygon:
        if type(test) is Point:
            return test.within(container)
        elif type(test) is Polygon:
            return container.contains(test)
    else:
        raise NotImplementedError


# generate obstacles in config.py
def generate_indoor_obstacles(config, wall_pts):

    obstacles = {}
    num_obstacles_left = config.obstacle.static.num

    # attempt to generate obstacles
    while num_obstacles_left > 0:
        # describes current obstacle
        descriptor = {"pts": None}

        radius = np.random.uniform(
            min(config.obstacle.static.size_range),
            max(config.obstacle.static.size_range),
        )
        # proposed position
        pt = Point(rand_world_pt(config), rand_world_pt(config))
        curr_obstacle = make_shapely_ellipse(radius, [pt.x, pt.y])

        # add offsets to ensure obstacle within simulation world
        x_offset, y_offset = generate_offset([pt.x, pt.y], radius)
        pt.x += x_offset
        pt.y += y_offset

        collision = False
        # check if point is outside confined space
        if config.walls.enable and not inside(pt, wall_pts):
            # technically not a collision but just a check fail
            collision = True
            continue

        descriptor["points"] = list(curr_obstacle.exterior.coords)
        descriptor["center"] = tuple(pt.x, pt.y)
        descriptor["radius"] = radius
        if len(obstacles) == 0:
            # add first obstacle
            if not collision:
                # 0 indexed
                id_ = config.obstacle.static.num - num_obstacles_left
                num_obstacles_left -= 1
                obstacles[id_] = descriptor
        else:
            # check proposed position for collision with any existing obstacles
            for obs_id, obs_desc in obstacles.items():
                centroid_dist = vec_norm([pt.x, pt.y], obstacles)
                # for circlular obstacles
                min_dist = radius + obs_desc["radius"]

                # collision occurred between proposed x,y and already existing obstacles
                # OR obstacle separation is too small for largest pedestrian
                if centroid_dist < min_dist:
                    collision = True
                    break

            if not collision:
                id_ = config.obstacle.static.num - num_obstacles_left
                num_obstacles_left -= 1
                obstacles[id_] = descriptor

    return obstacles
