import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import shapely.wkt
from crowd_sim.envs.utils.helper import (
    make_shapely_ellipse,
    make_shapely_polygon,
    vec_norm,
)
from crowd_sim.envs.utils.lidar_to_grid_map import generate_ray_casting_grid_map
from matplotlib import patches
from shapely.affinity import rotate
from shapely.geometry import LineString, Point


class LidarSensor:
    # specify rounding to N decimal places
    NUM_DP = 3

    def __init__(self, cfg: dict):
        # self.CHANNELS = 16  # (UNUSED)
        # self.ROTATION_RATE = 1  # (UNUSED)
        self.SENSOR_POS = [0, 0]  # center of the robot
        self.SENSOR_HEADING = 0.0
        self.MAX_DIST = cfg["max_range"]
        self.ANGLES = np.linspace(0, 360, cfg["num_spacings"] + 1)

        self.obstacles = []
        self.mpl_dict = {}
        self.collision_pts = []  # absolute x,y where beams end

    def parse_obstacles(
        self,
        obstacle_pts: list,
        mode="",
    ):
        if "walls" in mode:
            # create polygon representing walls
            self.obstacles.append(make_shapely_polygon(obstacle_pts))
        elif "agents" in mode:
            # assumes a list of lists
            # format: (pos x, pos y, radius)
            for agent in obstacle_pts:
                self.obstacles.append(make_shapely_ellipse(agent[-1], agent[:2]))
        else:
            pass

    def update_sensor(self, xy_pos=None, heading=None):
        if xy_pos is not None:
            self.SENSOR_POS = [xy_pos[0], xy_pos[1]]
        if heading is not None:
            self.SENSOR_HEADING = heading

        self._lidar_spin()

    def _lidar_spin(self):
        mpl_lines = []
        for _, angle in enumerate(self.ANGLES + self.SENSOR_HEADING):
            line = LineString(
                [
                    self.SENSOR_POS,
                    (self.MAX_DIST + self.SENSOR_POS[0], self.SENSOR_POS[1]),
                ]
            )
            rot_line = rotate(line, angle, origin=self.SENSOR_POS)
            # check for any intersects
            idx = -1
            for i, obstacle in enumerate(self.obstacles):
                any_intersect = rot_line.intersects(obstacle)
                if any_intersect:
                    idx = i
            # get specific points
            if idx != -1:
                # oh my god this is jank...
                chopped_line = shapely.wkt.loads(
                    shapely.wkt.dumps(
                        rot_line.intersection(self.obstacles[idx]),
                        rounding_precision=self.NUM_DP,
                    )
                )
                # access line intersection start and end points
                start, end = chopped_line.boundary

                if vec_norm(start.coords[0], self.SENSOR_POS) > 0:
                    # multiple intersection points
                    x_data = (self.SENSOR_POS[0], start.coords[0][0])
                    y_data = (self.SENSOR_POS[1], start.coords[0][1])
                else:
                    # if only wall, start = self.SENSOR_POS
                    x_data = (start.coords[0][0], end.coords[0][0])
                    y_data = (start.coords[0][1], end.coords[0][1])
                mpl_lines.append(
                    mlines.Line2D(
                        x_data,
                        y_data,
                        solid_capstyle="round",
                        marker="o",
                        markersize=2,
                        color="r",
                    )
                )
                col_pt = (x_data[1], y_data[1])
                self.collision_pts.append(col_pt)
        self.mpl_dict["lines"] = mpl_lines

    def get_mpl_dict(self):
        return self.mpl_dict


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 7))
    val = 20
    ax.set_xlim(-val * 1.5 / 2, val * 1.5 / 2)
    ax.set_ylim(-val * 1.5 / 2, val * 1.5 / 2)
    ax.set_xlabel("x(m)", fontsize=16)
    ax.set_ylabel("y(m)", fontsize=16)

    cfg = {"max_range": np.sqrt(2) * val / 2, "num_spacings": 145}
    lidar = LidarSensor(cfg)
    t = val / 4
    wall_pts = [(-t, -t), (t, -t), (t, t), (-t, t)]
    lidar.parse_obstacles(wall_pts, "walls")
    agent = [(1, 1, 0.5), (-2, -2, 0.5)]
    for a in agent:
        agent_circle = patches.Circle(a[:2], a[-1], color="b")
        ax.add_patch(agent_circle)
    lidar.parse_obstacles(agent, "agents")

    pos, heading = (0, 0), 0  # x,y, degrees
    lidar.update_sensor(pos, heading)

    lines = lidar.get_mpl_dict()["lines"]
    for l in lines:
        ax.add_line(l)

    end_pts = lidar.collision_pts
    if end_pts[0] != end_pts[-1]:
        end_pts.append(end_pts[0])
    polygon = patches.Polygon(xy=end_pts, fill=False, color="g")
    ax.add_patch(polygon)
    ax.axis("on")

    # extract image data from mpl plot
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    w, h = fig.canvas.get_width_height()
    data = data.reshape(h, w, 3)
    # from hough import test_opencv, test_skimage
    # test_skimage(data)
    # test_opencv(data)
    plt.show()
