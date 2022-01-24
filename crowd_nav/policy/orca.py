import numpy as np
import rvo2
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.rectangles import Rectangle


class ORCA(Policy):
    def __init__(self, config, obstacles_dict: dict = None):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__(config)
        self.name = "ORCA"
        self.max_neighbors = None
        self.radius = None
        # the ego agent assumes that all other agents have this max speed
        self.max_speed = config.humans.v_pref
        self.sim = None
        self.obstacles = obstacles_dict if obstacles_dict else None

        # TODO: fix wall boundaries
        if config.obstacle.walls.enable:
            world_size = config.sim.square_width
            # misleading because it's not a rectangle
            self.polygon_vertices = list(
                Rectangle(world_size, world_size)._rect.exterior.coords
            )

    def _parse_obstacles(self):
        """Parse indoor obstacles into RVO2 simulator"""
        if self.obstacles:
            num_obstacles = len(self.obstacles)
            if num_obstacles > 0:
                # loop through obstacle descriptors, see obstacles.py for NESTED DICT keys
                for _, obst_desc in self.obstacles.items():
                    assert type(obst_desc) is dict
                    # method requires obstacle vertices in counter-clockwise
                    self.sim.addObstacle(obst_desc.get("points"))
                # must be called
                self.sim.processObstacles()
            else:
                print("ORCA: Obstacles dictionary empty!")

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = state.self_state
        # max number of humans = current number of humans
        self.max_neighbors = len(state.human_states)
        self.radius = state.self_state.radius
        params = (
            self.config.orca.neighbor_dist,
            self.max_neighbors,
            self.config.orca.time_horizon,
            self.config.orca.time_horizon_obst,
        )
        if (
            self.sim is not None
            and self.sim.getNumAgents() != len(state.human_states) + 1
        ):
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(
                self.config.env.time_step, *params, self.radius, self.max_speed
            )
            self.sim.addAgent(
                self_state.position,
                *params,
                self_state.radius + 0.01 + self.config.orca.safety_space,
                self_state.v_pref,
                self_state.velocity
            )
            for human_state in state.human_states:
                self.sim.addAgent(
                    human_state.position,
                    *params,
                    human_state.radius + 0.01 + self.config.orca.safety_space,
                    self.max_speed,
                    human_state.velocity
                )
            self._parse_obstacles()
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array(
            (self_state.gx - self_state.px, self_state.gy - self_state.py)
        )
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action
