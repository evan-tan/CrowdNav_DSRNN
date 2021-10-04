import time
from typing import List

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
from crowd_sim.envs.utils.helper import (
    make_mpl_line,
    make_shapely_ellipse,
    make_shapely_polygon,
    vec_norm,
)
from matplotlib import patches
from shapely import geometry
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from sklearn.preprocessing import minmax_scale


def replace_poly_with_point(shape: Polygon, angles) -> Point:
    """Replace Polygon with closest Point

    :param shape: Shapely object
    :type shape: Polygon
    :return: return Point closest to sensor origin
    :rtype: Point
    """
    if np.max(angles) > (2 * np.pi):
        angles = np.deg2rad(angles)
    radius = abs(abs(shape.bounds[2]) - abs(shape.bounds[0])) / 2
    # get centroid
    cx, cy = shape.centroid.x, shape.centroid.y
    # get point closest to origin
    heading = np.arctan2(cy, cx)
    # map to 0-360
    heading = (heading + 2 * np.pi) % (2 * np.pi)
    dx = radius * np.cos(heading)
    dy = radius * np.sin(heading)
    # angle difference between discrete lidar beam intercept with polygon, and the point on the polygon (adjust based on curvature)
    idx = np.argmin(abs(angles - heading))
    gamma = angles[idx] - heading
    adj_x = radius * np.cos(heading + gamma)
    adj_y = radius * np.sin(heading + gamma)
    x = cx - dx + adj_x
    y = cy - dy + adj_y

    # if heading >= 0 and heading <= np.pi / 2:
    #     x = cx - dx
    #     y = cy - dy
    # elif heading > np.pi / 2 and heading <= np.pi:
    #     x = cx + dx
    #     y = cy - dy
    # elif heading > np.pi and heading <= 1.5 * np.pi:
    #     x = cx + dx
    #     y = cy + dy
    # else:
    #     x = cx - dx
    #     y = cy + dy

    return Point(x, y)


agent = [
    (1, 1, 0.5),
    (2, 2, 0.5),
    (-4, -3, 0.5),
    (-3, -3, 0.3),
    (-3, -4, 0.2),
    (-0.5, 0.5, 0.4),
    (5, -5, 0.3),
]

fig, ax = plt.subplots(figsize=(7, 7))
val = 20 / 2 + 5
ax.set_xlim(-val, val)
ax.set_ylim(-val, val)
ax.set_xlabel("x(m)", fontsize=16)
ax.set_ylabel("y(m)", fontsize=16)
for a in agent:
    agent_circle = patches.Circle(a[:2], a[-1], color="b")
    ax.add_patch(agent_circle)

t = 20 / 2
wall_pts = [(-t, -t), (t, -t), (t, t), (-t, t)]
if wall_pts[-1] != wall_pts[0]:
    wall_pts.append(wall_pts[0])
# decompose wall polygon into a series on LineStrings
cfg = {"max_range": 11, "num_spacings": 180}
angles = np.linspace(0, 360, 180)
n_iter = int(1e2)
start = time.time()
for _ in range(n_iter):
    lines = []
    obst_max_range = []
    max_dist = 5
    obst = []  # store all obstacles
    for angle in np.linspace(0, 360, 180):
        line = LineString(
            [
                [0, 0],
                (max_dist + 0, 0),
            ]
        )
        rot_line = rotate(line, angle, origin=[0, 0])
        lines.append(rot_line)
        _, end = rot_line.boundary
        obst.append(Point(end.x, end.y))

    gp_lines = gp.GeoDataFrame({"geometry": lines})

    for i in range(len(wall_pts) - 1):
        obst.append(LineString([wall_pts[i], wall_pts[i + 1]]))

    for pt in agent:
        obst.append(make_shapely_ellipse(pt[-1], pt[:2]))

    gp_obst = gp.GeoDataFrame({"geometry": obst})

    col_name = "index_beam"
    df = gp.sjoin(
        left_df=gp_obst,
        right_df=gp_lines,
        how="inner",
        rsuffix=col_name.replace("index_", ""),  # column becomes: index_beam
        predicate="intersects",
    )
    df = df.sort_values(col_name).reset_index(drop=True)

    poly_indices = df.index[df["geometry"].type == "Polygon"]
    df["geometry"].iloc[poly_indices] = (
        df["geometry"]
        .iloc[poly_indices]
        .apply(lambda x: replace_poly_with_point(x, angles))
    )
    # since everything is reduced to points, we can safely use centroids
    df["dist_to_centroid"] = df["geometry"].distance(Point(0, 0))
    # get closest geometry for each lidar beam
    df = (
        df.sort_values("dist_to_centroid", ascending=True)
        .drop_duplicates([col_name])
        .sort_values(col_name)
        .reset_index(drop=True)
    )


elapsed = time.time() - start
print(f"{n_iter} iterations took {elapsed:4f}s")

df.to_csv("DATAFRAME.csv")


distances = df["dist_to_centroid"].to_numpy()
x_data = distances * np.cos(np.deg2rad(angles))
y_data = distances * np.sin(np.deg2rad(angles))
mpl_lines = []
for i in range(x_data.shape[0]):
    ax.add_line(make_mpl_line((0, x_data[i]), (0, y_data[i])))
    # print(x_data[i], y_data[i])


plt.show()
# intersections.to_csv("final.csv")
# # geometry intersection for each lidar beam
# df_sorted = intersections["geometry"].reset_index(drop=True)
# # compute line that defines where they intersect
# # see https://geopandas.org/docs/reference/api/geopandas.GeoSeries.intersection.html
# df_final = gp_lines.intersection(df_sorted, align=True)
