import random

import numpy as np
from crowd_sim.envs.utils.helper import ang_diff, vec_norm
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


# NOTE: square_width is essentially the radius of the world environment (square)
# determine if we want points to wrap around
def generate_polygon(
    polygon_shape: str, polygon_scale: float, square_width: float, wrap: bool = False
):

    polygon_vertices = []

    if "square" in polygon_shape.lower():
        val = polygon_scale * square_width
        # clockwise direction
        # [-val, -val], [-val, val], [val, val], [val, -val]
        # counter clockwise direction
        polygon_vertices = [[-val, -val], [val, -val], [val, val], [-val, val]]
    else:
        pass

    if wrap:
        if polygon_vertices[-1] != polygon_vertices[0]:
            polygon_vertices.append(polygon_vertices[0])
        # polygon_vertices = np.array(polygon_vertices)
    else:
        pass

    return polygon_vertices


# generate ORCA boundary obstacles
def generate_boundary(polygon_vertices, thickness):
    # wrap points around
    if polygon_vertices[-1] != polygon_vertices[0]:
        polygon_vertices.append(polygon_vertices[0])

    for i in range(len(polygon_vertices) - 1):
        ptA = polygon_vertices[i]
        ptB = polygon_vertices[i + 1]

        length = vec_norm(ptA, ptB)
        angle = ang_diff(ptA, ptB)


# check if a particular point is inside polygon
# True, if inside polygon, False otherwise
def inside_polygon(polygon_vertices, point):
    # don't wrap values around list
    if polygon_vertices[-1] == polygon_vertices[0]:
        polygon_vertices = polygon_vertices[:-1]

    # create shapely objects
    polygon_shape = Polygon(polygon_vertices)
    point_shape = Point(point)

    return point_shape.within(polygon_shape)


# generate obstacles in config.py
def generate_obstacle_points(config):
    all_obstacles = []
    radii = []
    centers = []
    boundary_vertices = []
    num_obstacles_left = config.sim.obstacle_num

    if config.sim.confined_space:
        boundary_vertices = generate_polygon(
            config.sim.polygon_shape, config.sim.polygon_scale, config.sim.square_width
        )

    # attempt to generate obstacles
    while num_obstacles_left > 0:
        radius = np.random.uniform(
            min(config.sim.obstacle_size_range),
            max(config.sim.obstacle_size_range),
        )

        chosen_shape = random.choice(config.sim.obstacle_shape)
        curr_obstacle = generate_polygon(chosen_shape, 0.5, radius)
        collision = False

        # place obstacle at random point
        x = (np.random.random() - 0.5) * 2
        y = (np.random.random() - 0.5) * 2
        if config.sim.confined_space:
            x *= config.sim.polygon_scale * config.sim.square_width
            y *= config.sim.polygon_scale * config.sim.square_width
        else:
            x *= config.sim.square_width
            y *= config.sim.square_width

        x_offset, y_offset = generate_offset([x, y], radius)
        # add offsets to ensure obstacle within simulation world
        x += x_offset
        y += y_offset

        # check if point is outside confined space
        if config.sim.confined_space and not inside_polygon(boundary_vertices, [x, y]):
            # technically not a collision but just a check fail
            collision = True
            continue

        if len(centers) and len(radii) == 0:
            # add first obstacle
            if not collision:
                # offset obstacle to correct position
                for pt in curr_obstacle:
                    pt[0] += x
                    pt[1] += y
                all_obstacles.append(curr_obstacle)
                radii.append(radius)
                centers.append([x, y])
                num_obstacles_left -= 1

        else:
            # check proposed position for collision with any existing obstacles
            for i in range(len(radii)):
                centroid_dist = vec_norm([x, y], centers[i])
                if "square" in chosen_shape:
                    # maximum distance between 2 squares
                    min_dist = np.sqrt(2) * (radii[i] + radius)
                else:
                    # for circlular obstacles
                    min_dist = radii[i] + radius

                # collision occurred between proposed x,y and already existing obstacles
                # OR obstacle separation is too small for largest pedestrian
                if centroid_dist < min_dist:
                    collision = True
                    break
            # centroid_dist < max(config.humans.radii_range)

            if not collision:
                # offset obstacle to correct position
                for pt in curr_obstacle:
                    pt[0] += x
                    pt[1] += y
                all_obstacles.append(curr_obstacle)
                radii.append(radius)
                centers.append([x, y])
                num_obstacles_left -= 1

    return all_obstacles


def generate_static_obstacles():
    pass


if __name__ == "__main__":
    print(generate_polygon("square", 0.5, 20, False))
