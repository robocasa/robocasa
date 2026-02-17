from robocasa.models.fixtures import Fixture


class Dishwasher(Fixture):
    """
    Dishwasher fixture class
    """

    def __init__(
        self,
        xml="fixtures/dishwashers/Dishwasher031",
        name="dishwasher",
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

        self._door = 0.0
        self._rack = 0.0

        # Initialize joint names dictionary
        joint_prefix = self._get_joint_prefix()
        self._joint_names = {
            "door": f"{joint_prefix}door_joint",
            "rack": f"{joint_prefix}rack1_joint",
        }

    def get_reset_region_names(self):
        return ("rack0", "rack1")

    def _get_joint_prefix(self):
        return f"{self.naming_prefix}"

    def slide_rack(self, env, value=1.0):
        """
        Pulls the specified dishwasher rack out completely.

        Args:
            value (float): Value to set the rack to
        """

        door_pos = self.get_joint_state(env, [self._joint_names["door"]])[
            self._joint_names["door"]
        ]
        if door_pos <= 0.99:
            self.open_door(env)

        self.set_joint_state(
            env=env, min=value, max=value, joint_names=[self._joint_names["rack"]]
        )

    def update_state(self, env):
        """
        Updates internal state variables from the simulation environment.
        """
        for name, jn in self._joint_names.items():
            if jn in env.sim.model.joint_names:
                state = self.get_joint_state(env, [jn])[jn]
                setattr(self, f"_{name}", state)

    def check_rack_contact(self, env, obj_name):
        """
        Checks whether the specified object is in contact with the top rack.
        """
        joint_name = self._joint_names["rack"]
        joint_id = env.sim.model.joint_name2id(
            joint_name
        )  # don't use get_joint_qpos_addr, causes errors
        body_id = env.sim.model.jnt_bodyid[joint_id]

        rack_geoms = [
            env.sim.model.geom_id2name(i)
            for i in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[i] == body_id
        ]

        obj_body_id = env.obj_body_id[obj_name]
        item_geoms = [
            env.sim.model.geom_id2name(gid)
            for gid in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[gid] == obj_body_id
        ]

        return env.check_contact(rack_geoms, item_geoms)

    def get_state(self, env):
        """
        Returns the current state of the dishwasher as a dictionary.
        """
        st = {}
        for name, jn in self._joint_names.items():
            if jn in env.sim.model.joint_names:
                st[name] = getattr(self, f"_{name}", None)
        return st

    @property
    def nat_lang(self):
        return "dishwasher"
