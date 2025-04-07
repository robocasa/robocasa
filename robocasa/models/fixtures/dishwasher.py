from robocasa.models.fixtures import Fixture
from robosuite.utils.mjcf_utils import (
    string_to_array,
    find_elements,
)


class Dishwasher(Fixture):
    """
    Dishwasher fixture class
    """

    def __init__(
        self,
        xml="fixtures/appliances/dishwashers/Dishwasher031",
        name="dishwasher",
        *args,
        **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self._dishwasher_door_joint_info = dict()
        joint_elems = find_elements(
            root=self.worldbody, tags="joint", return_first=False
        )
        for elem in joint_elems:
            elem_name = elem.get("name")
            if elem_name is None:
                continue
            # strip out name prefix
            stripped_name = elem_name[len(self.name) + 1 :]
            if "door" in stripped_name:
                self._dishwasher_door_joint_info[elem_name] = dict(
                    range=string_to_array(elem.get("range")),
                )

    def set_door_state(self, min, max, env):
        """
        Sets how open the door is. Chooses a random amount between min and max.
        Min and max are percentages of how open the door is
        Args:
            min (float): minimum percentage of how open the door is
            max (float): maximum percentage of how open the door is
            env (MujocoEnv): environment
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        rng = None or env.rng

        for joint_name, info in self._dishwasher_door_joint_info.items():
            joint_min, joint_max = info["range"]
            desired_min = joint_min + (joint_max - joint_min) * min
            desired_max = joint_min + (joint_max - joint_min) * max
            env.sim.data.set_joint_qpos(
                joint_name,
                rng.uniform(desired_min, desired_max),
            )

    def get_reset_region_names(self):
        return ("rack_bottom", "rack_top")

    @property
    def nat_lang(self):
        return "dishwasher"
