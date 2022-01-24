import random
from typing import Dict

import numpy as np
from crowd_sim.envs.utils.helper import (
    PointXY,
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
def inside(test, container):
    """
    Test whether an object is inside another currently supports:
    Point in Polygon test
    Polygon in Polygon test

    returns True, if inside, False otherwise
    """
    # jank type conversions
    if type(container) is list:
        # convert list of points into a polygon
        container = Polygon(container)
    if type(test) != Point:
        if type(test) is PointXY:
            test = Point(PointXY.x, PointXY.y)
        elif type(test) is list:
            test = Point(test)

    # logic
    if type(container) is Polygon:
        if type(test) is Point:
            return test.within(container)
        elif type(test) is Polygon:
            return container.contains(test)
    else:
        raise NotImplementedError


# generate obstacles in config.py
def generate_indoor_obstacles(config, wall_pts) -> Dict:

    obstacles = {}
    num_obstacles_left = config.obstacle.static.num

    while num_obstacles_left > 0:
        # describes current obstacle
        descriptor = {}

        radius = np.random.uniform(
            min(config.obstacle.static.size_range),
            max(config.obstacle.static.size_range),
        )
        # proposed position
        pt = PointXY(rand_world_pt(config), rand_world_pt(config))
        # for example, using a radius of 1.0
        # downsampling reduces from 66 points -> 9 points
        curr_obstacle = make_shapely_ellipse(radius, [pt.x, pt.y], downsample=True)

        # add offsets to ensure obstacle within simulation world
        x_offset, y_offset = generate_offset([pt.x, pt.y], radius)
        pt.x += x_offset
        pt.y += y_offset

        collision = False
        # check if point is outside confined space
        if config.obstacle.walls.enable and not inside(pt, wall_pts):
            # technically not a collision but just a check fail
            collision = True
            continue

        # when parsing into ORCA, obstacle vertices should be counter-clockwise
        descriptor["points"] = list(curr_obstacle.exterior.coords)[::-1]
        descriptor["center"] = tuple([pt.x, pt.y])
        descriptor["radius"] = radius

        # check proposed position for collision with any EXISTING obstacles
        for obs_id, obs_desc in obstacles.items():
            centroid_dist = vec_norm([pt.x, pt.y], obs_desc["center"])
            # assume circlular obstacles
            min_dist = radius + obs_desc["radius"]

            # collision occurred between proposed x,y and already existing obstacles
            # OR obstacle separation is too small for largest pedestrian
            if centroid_dist < min_dist:
                collision = True
                print(f"obstacles: {pt.x=}, {pt.y=} collided with Obstacle{obs_id}!")
                break

        if not collision:
            id_ = config.obstacle.static.num - num_obstacles_left
            num_obstacles_left -= 1
            obstacles[id_] = descriptor

    return obstacles
