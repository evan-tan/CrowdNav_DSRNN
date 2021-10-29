import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from crowd_sim.envs.utils.helper import (
    make_mpl_line,
    make_shapely_ellipse,
    make_shapely_polygon,
    vec_norm,
)
from matplotlib import patches
from shapely.affinity import rotate
from shapely.geometry import LineString
from shapely.strtree import STRtree
from sklearn.preprocessing import minmax_scale


# TODO: use cuSpatial for GPU
class LidarSensor:
    """LiDAR Sensor to create occupancy maps using both dynamic and static obstacles, consideres a maximum sensor range and computes the angles and distances
    NOTE: distances are min/max scaled according to sensor range!!!"""

    def __init__(self, cfg: dict):
        self.sensor_pos = [0, 0]  # center of the robot
        self.sensor_heading = 0.0
        self.MAX_DIST = cfg["max_range"]
        self.lidar_angles = np.linspace(0, 360, cfg["num_beams"])

        self.obstacles = {"walls": None, "agents": None}
        self.agent_attrs = []  # (x,y,radius)
        self.lidar_end_pts = None
        self.mpl_lines = []  # matplotlib lines for visualization

    def parse_obstacles(
        self,
        obstacle_pts: list,
        mode="",
    ):
        """Parse dynamic/static obstacles for collision checking when updating sensor"""
        if "walls" in mode:
            # create polygon representing walls
            self.obstacles[mode] = make_shapely_polygon(obstacle_pts)

        elif "agents" in mode:
            assert self.obstacles["walls"] is not None
            # NOTE: assumes a list of lists
            # format: (pos x, pos y, radius)
            tmp_list = []
            for agent in obstacle_pts:
                tmp_list.append(make_shapely_ellipse(agent[-1], agent[:2]))
            self.obstacles[mode] = tmp_list
            self.agent_attrs = obstacle_pts
        else:
            raise NotImplementedError

    def update_sensor(self, xy_pos: List[float] = None, heading: float = None):
        """Update sensor according to position and heading

        :param xy_pos: Robot position, defaults to None
        :type xy_pos: List[float], optional
        :param heading: Robot heading, defaults to None
        :type heading: float, optional
        """
        if xy_pos is not None:
            self.sensor_pos = [xy_pos[0], xy_pos[1]]
        if heading is not None:
            self.sensor_heading = heading

    def sensor_spin(self, viz=False, normalize=True):
        """Get all end points for each lidar beam

        :param viz: Create matplotlib lines, defaults to False
        :type viz: bool, optional
        :return: Angles and distances
        :rtype: numpy.ndarray
        """
        if viz:
            mpl_lines = []

        end_pts = []
        # rotate all lidar beams
        rot_lidar_angles = self.lidar_angles + self.sensor_heading
        all_obstacles = [self.obstacles["walls"], *self.obstacles["agents"]]
        tree = STRtree(all_obstacles)
        for angle in rot_lidar_angles:
            # create beams and rotate to correct angle
            line = LineString(
                [
                    self.sensor_pos,
                    (self.MAX_DIST + self.sensor_pos[0], self.sensor_pos[1]),
                ]
            )
            rot_line = rotate(line, angle, origin=self.sensor_pos)
            # index 1 for end, index 0 for start of line
            rot_end = rot_line.boundary[1].coords[0]
            result = tree.query(rot_line)
            # check for any intersects between current beam and ALL obstacles
            for obstacle in result:
                tmp_line = rot_line.intersection(obstacle)
                # no intersect
                if tmp_line.is_empty:
                    continue
                else:
                    # extract end point
                    tmp_end = tmp_line.boundary[1].coords[0]
                    # if multiple intersections, always consider closest line
                    if vec_norm(tmp_end, [0, 0]) < vec_norm(rot_end, [0, 0]):
                        # intersection_line = tmp_line
                        rot_line = tmp_line

            # access line intersection information
            start, end = rot_line.boundary
            if vec_norm(start.coords[0], self.sensor_pos) > 0:
                # multiple intersection points
                # if collide with obstacle,
                # start/end = closer/further parts of polygon
                x_data = (self.sensor_pos[0], start.coords[0][0])
                y_data = (self.sensor_pos[1], start.coords[0][1])
                end_pt = start.coords[0]
            else:
                # if no collision
                # start = sensor pos, end = point @ max range!
                x_data = (start.coords[0][0], end.coords[0][0])
                y_data = (start.coords[0][1], end.coords[0][1])
                end_pt = end.coords[0]
            # store all end points
            end_pts.append(end_pt)

            # use this if you're lazy to create lines
            if viz:
                mpl_lines.append(make_mpl_line(x_data, y_data))
        if viz:
            self.mpl_lines = mpl_lines

        # convert to np array
        distances = np.linalg.norm(np.array(end_pts), axis=1).squeeze()
        if normalize:
            minmax_scale(distances, feature_range=(0, 1), copy=False)
        self.lidar_end_pts = end_pts

        return rot_lidar_angles, distances


if __name__ == "__main__":
    # DEBUGGING
    fig, ax = plt.subplots(figsize=(7, 7))
    val = 20 / 2 + 5
    ax.set_xlim(-val, val)
    ax.set_ylim(-val, val)
    ax.set_xlabel("x(m)", fontsize=16)
    ax.set_ylabel("y(m)", fontsize=16)

    cfg = {"max_range": 11, "num_beams": 180}
    lidar = LidarSensor(cfg)

    # construct a 20x20 box, centered @ origin
    # max dist from lidar = sqrt(2) * 10
    t = 20 / 2
    wall_pts = [(-t, -t), (t, -t), (t, t), (-t, t)]

    # # create some dummy agents
    agent = [
        (1, 1, 0.5),
        (2, 2, 0.5),
        (-4, -3, 0.5),
        (-3, -3, 0.3),
        (-3, -4, 0.2),
        (-0.5, 0.5, 0.4),
        (5, -5, 0.3),
    ]
    for a in agent:
        agent_circle = patches.Circle(a[:2], a[-1], color="b")
        ax.add_patch(agent_circle)

    pos, heading = (0, 0), 0  # x,y, degrees

    n_iter = 1e2
    start = time.time()
    lidar.parse_obstacles(wall_pts, "walls")
    for i in range(int(n_iter)):
        lidar.parse_obstacles(agent, "agents")
        lidar.update_sensor(pos, heading)
        angles, distances = lidar.sensor_spin(viz=True)

    elapsed = time.time() - start
    print(f"{n_iter} iterations took {elapsed:4f}s")

    for line_ in lidar.mpl_lines:
        ax.add_line(line_)

    fig2, ax2 = plt.subplots()
    ax2.plot(angles, distances)
    ax2.set_xlabel("Lidar Angle(degrees)")
    ax2.set_ylabel("Normalized Distance")
    ax2.set_title("Lidar Data in Polar Coordinates with Distance Min/Max Normalization")
    ax.annotate(
        "Max Sensor Range = 11m",
        xy=(210, 25),
        xycoords="axes points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", fc="w"),
    )
    ax2.annotate(
        "Max Sensor Range = 11m",
        xy=(210, 75),
        xycoords="axes points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", fc="w"),
    )
    # extract image data from mpl plot
    # from hough import test_opencv, test_skimage
    # fig.canvas.draw()
    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    # w, h = fig.canvas.get_width_height()
    # data = data.reshape(h, w, 3)
    # test_skimage(data)
    # test_opencv(data)

    plt.show()
