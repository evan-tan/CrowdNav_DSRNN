# %%
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import shapely.wkt
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, Polygon


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

        self.walls = None
        self.lasers = []
        self.collision_pts = []  # wherethe lidar collides into obstacles

    def parse_obstacles(
        self,
        obstacles: list,
        mode="",
    ):
        if "walls" in mode:
            # ensure that walls loop around
            if obstacles[0] != obstacles[-1]:
                obstacles.append(obstacles[0])
            # create polygon representing walls
            self.walls = Polygon(obstacles)
        elif "agents" in mode:
            pass
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
            # check if line should be truncated due to obstacles
            # oh my god this is jank...
            chopped_line = shapely.wkt.loads(
                shapely.wkt.dumps(
                    rot_line.intersection(self.walls), rounding_precision=self.NUM_DP
                )
            )

            # access line start and end points
            start, end = chopped_line.boundary
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
            self.mpl_lines = mpl_lines
            self.lasers.append(chopped_line)  # store all LineStrings
            self.collision_pts.append(end.coords[0][:])

    def get_mpl_lines(self):
        return self.mpl_lines


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 7))
    val = 20
    ax.set_xlim(-val * 1.5 / 2, val * 1.5 / 2)
    ax.set_ylim(-val * 1.5 / 2, val * 1.5 / 2)
    ax.set_xlabel("x(m)", fontsize=16)
    ax.set_ylabel("y(m)", fontsize=16)

    cfg = {"max_range": np.sqrt(2) * val / 2, "num_spacings": 135}
    lidar = LidarSensor(cfg)
    t = val / 4
    wall_pts = [(-t, -t), (t, -t), (t, t), (-t, t)]
    lidar.parse_obstacles(wall_pts, "walls")
    pos, heading = (2, 1), 45  # x,y, degrees
    lidar.update_sensor(pos, heading)

    lines = lidar.get_mpl_lines()
    for _, l in enumerate(lines):
        ax.add_line(l)

    plt.show()
