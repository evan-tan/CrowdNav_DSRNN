import numpy as np

# A Python3 program to find if 2 given line segments intersect or not
# This code is contributed by Ansh Riyal
# SOURCE: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y


def is_on_segment(p, q, r):
    # very very jank indexing
    if (
        (q[0] <= max(p[0], r[0]))
        and (q[0] >= min(p[0], r[0]))
        and (q[1] <= max(p[1], r[1]))
        and (q[1] >= min(p[1], r[1]))
    ):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if val > 0:
        return 1  # Clockwise orientation
    elif val < 0:
        return 2  # Counterclockwise orientation
    else:
        return 0  # Collinear orientation


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def check_intersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and is_on_segment(p1, p2, q1):
        return True
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and is_on_segment(p1, q2, q1):
        return True
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and is_on_segment(p2, p1, q2):
        return True
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and is_on_segment(p2, q1, q2):
        return True
    # If none of the cases
    return False


# SOURCE: https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
def get_intersect(p1, q1, p2, q2):
    """
    Returns the point of intersection of the lines passing through p1,q1 and p2,q2.
    p1: [x, y] a point on the FIRST LINE
    q1: [x, y] another point on the FIRST LINE
    p2: [x, y] a point on the SECOND LINE
    q2: [x, y] another point on the SECOND LINE
    """

    is_intersecting = check_intersect(p1, q1, p2, q2)
    intersect_pt = None
    if is_intersecting:
        s = np.vstack([p1, q1, p2, q2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z != 0:
            intersect_pt = [x / z, y / z]
    return intersect_pt


if __name__ == "__main__":
    # Driver program to test above functions:
    p1 = [1, 1]
    q1 = [10, 1]
    p2 = [1, 2]
    q2 = [10, 2]
    print(get_intersect(p1, q1, p2, q2))

    p1 = [10, 0]
    q1 = [0, 10]
    p2 = [0, 0]
    q2 = [10, 10]
    print(get_intersect(p1, q1, p2, q2))

    p1 = [-5, -5]
    q1 = [0, 0]
    p2 = [1, 1]
    q2 = [10, 10]
    print(get_intersect(p1, q1, p2, q2))
