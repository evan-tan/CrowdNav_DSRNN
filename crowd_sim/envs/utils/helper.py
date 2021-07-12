import shapely.geometry


# https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
class VelocityRectangle:
    def __init__(self, agent):
        self.agent = agent
        assert agent is not None
        self._create_rect()

    def _create_rect(self):
        # scaling factor for rect_len, see Kevin's paper
        k = 1
        rlength = k * (self.agent.vx ** 2 + self.agent.vy ** 2) ** 0.5
        rwidth = 2 * self.agent.radius
        heading = self.agent.theta

        # create rectangle at 0,0 first, bottom edge touching horizontal axis
        origin_rect = shapely.geometry.box(-rwidth / 2, 0, rwidth / 2, rlength)
        # rotate ABOUT (0,0) based on agent heading
        rot_rect = shapely.affinity.rotate(origin_rect, angle=heading, origin=(0, 0))
        # translate to get final rect
        self.vec_rect = shapely.affinity.translate(
            rot_rect, self.agent.px, self.agent.py
        )

    def intersects(self, VelocityRectangleObject):
        assert hasattr(VelocityRectangleObject, "vec_rect")
        return self.vec_rect.intersects(VelocityRectangleObject.vec_rect)
