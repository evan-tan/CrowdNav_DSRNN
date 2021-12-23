import numpy as np
import shapely


# https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
class Rectangle:
    """Rectangle base class.
    Creates rectangle centered at the origin (0,0).
    """

    def __init__(self, width, length):
        # create vertical rectangle with center at (0,0) with heading of +90deg
        self._rect = shapely.geometry.box(
            -width / 2, -length / 2, width / 2, length / 2
        )

    def _translate(self, dx, dy):
        self._rect = shapely.affinity.translate(self._rect, dx, dy)

    def _rotate(self, angle, rot_pt="center"):
        # rot_pt = "center", "centroid" or (x,y)
        self._rect = shapely.affinity.rotate(
            self._rect, angle, origin=rot_pt, use_radians=True
        )

    def intersects(self, RectangleObject):
        assert hasattr(RectangleObject, "_rect")
        return self._rect.intersects(RectangleObject._rect)


# https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
class VelocityRectangle(Rectangle):
    """Class to create VelocityRectangles that project agent's velocity by a factor to calculate social metric violations"""

    # k=3, how many time steps ahead to extend velocity rectangle by
    LENGTH_SCALE = 3
    WIDTH_SCALE = 1

    def __init__(self, agent=None):
        self._agent = agent
        assert agent is not None

        # in agent.py [self.px, self.py, self.vx, self.vy, self.radius]
        properties = self._agent.get_observable_state_list()
        vx, vy = properties[2:4]
        radius = properties[4]
        rwidth = 2 * radius * self.WIDTH_SCALE
        rlength = self.LENGTH_SCALE * (vx ** 2 + vy ** 2) ** 0.5

        # NOTE: DO NOT USE self._agent.theta as it is never updated
        # -90deg since origin_rect is ALREADY oriented at +90deg
        agent_heading = np.arctan2(vy, vx)
        dtheta = agent_heading - np.pi / 2
        x_os = self._agent.px + radius * np.cos(agent_heading)
        y_os = self._agent.py + radius * np.sin(agent_heading)

        super().__init__(rwidth, rlength)

        # make "bottom edge" of rectangle touch horizontal axis
        self._translate(0, rlength / 2)
        # rotate ABOUT (0,0) based on agent heading
        self._rotate(dtheta, (0, 0))
        # translate to get final rect
        self._translate(x_os, y_os)


class NormZoneRectangle(Rectangle):
    """Class to create NormZoneRectangle (aka social norm zones) as suggested in SARL/LM-SARL paper"""

    LENGTH_SCALE = 1.5
    WIDTH_SCALE = 1.5
    OFFSET = 0.6

    def __init__(self, agent, side="", norm="lhs"):
        self._agent = agent
        assert agent is not None
        assert "left" in side or "right" in side
        assert "lhs" in norm or "rhs" in norm

        # in agent.py [self.px, self.py, self.vx, self.vy, self.radius]
        properties = self._agent.get_observable_state_list()
        vx, vy = properties[2:4]
        radius = properties[4]
        rwidth = 2 * radius * self.WIDTH_SCALE
        rlength = self.LENGTH_SCALE * 1.2
        # DO NOT USE self._agent.theta as it is never updated
        # -90deg since origin_rect is ALREADY oriented at +90deg
        agent_heading = np.arctan2(vy, vx)
        dtheta = agent_heading - np.pi / 2
        x_os = self._agent.px + radius * np.cos(agent_heading)
        y_os = self._agent.py + radius * np.sin(agent_heading)

        super().__init__(rwidth, rlength)

        # default behaviour for rhs norm
        # make "bottom (short) edge" of rectangle touch horizontal axis, translate to left/right
        if "lhs" in norm:
            if "left" in side:
                # LHS of robot
                self._translate(-rwidth / 2, rlength / 2 + self.OFFSET)
            elif "right" in side:
                # RHS of robot and translated forward by 0.6m
                self._translate(rwidth / 2, rlength / 2)
        elif "rhs" in norm:
            if "left" in side:
                self._translate(-rwidth / 2, rlength / 2)
            elif "right" in side:
                self._translate(rwidth / 2, rlength / 2 + self.OFFSET)

        # rotate ABOUT (0,0) based on agent heading
        self._rotate(dtheta, (0, 0))
        # translate to get final rect
        self._translate(x_os, y_os)
