from robocasa.models.fixtures import Fixture
from robosuite.utils.mjcf_utils import find_elements, string_to_array


class Toaster(Fixture):
    """
    Toaster fixture class
    """

    def __init__(self, xml, name="toaster", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

        self._floor_geoms = {}
        for slot_name in ["slot_left", "slot_right"]:
            geom = find_elements(
                self.worldbody,
                "geom",
                {"name": f"{self.naming_prefix}{slot_name}_floor"},
            )
            if geom is not None:
                self._floor_geoms[slot_name] = geom

        self._num_steps_on = 0
        self._turned_on = False

    @property
    def nat_lang(self):
        return "toaster"

    def get_reset_regions(self, *args, **kwargs):
        """
        returns dictionary of reset regions, usually used when initialzing a mug under the coffee machine
        """
        reset_regions = {}
        for slot_name, geom in self._floor_geoms.items():
            geom_pos = string_to_array(geom.get("pos"))
            geom_size = string_to_array(geom.get("size"))

            reset_regions[slot_name] = {
                "offset": geom_pos + [0, 0, geom_size[2]],
                "size": (geom_size[0] * 2 * 2.5, geom_size[1] * 2.5),
            }
        return reset_regions

    def update_state(self, env):
        """
        Updates the burner flames of the stove based on the knob joint positions

        Args:
            env (MujocoEnv): environment
        """
        joint_name = "{}{}".format(self.naming_prefix, "Lever001_joint")
        if joint_name in env.sim.model.joint_names:
            joint_id = env.sim.model.joint_name2id(joint_name)
        else:
            return

        joint_qpos = env.sim.data.qpos[joint_id]

        if joint_qpos <= 0.040:
            self._turned_on = False
            self._num_steps_on = 0

        if self._turned_on is False and joint_qpos >= 0.084:
            self._turned_on = True
            self._num_steps_on = 0

        if self._turned_on:
            if self._num_steps_on > 100:
                self._turned_on = None
                self._num_steps_on = 0
            else:
                self._num_steps_on += 1

        if self._turned_on:
            env.sim.data.qpos[joint_id] = 0.085
