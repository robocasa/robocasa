from robocasa.environments.kitchen.kitchen import *


class FoodCleanup(Kitchen):
    """
    Food Cleanup: composite task for Clearing Table activity.

    Simulates the task of cleaning up various food items left on the counter.

    Steps:
        Pick the food items from the counter and place them in the cabinet.
        Then close the cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the
            food items are picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET_TOP, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab
        if "object_cfgs" in self._ep_meta:
            object_cfgs = self._ep_meta["object_cfgs"]
            self.num_food = len(
                [cfg for cfg in object_cfgs if cfg["name"].startswith("food")]
            )
        else:
            self.num_food = self.rng.choice([i for i in range(1, 4)])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        items = self.get_obj_lang("food0")
        for i in range(1, self.num_food):
            items += f", {self.get_obj_lang(f'food{i}')}"
        ep_meta[
            "lang"
        ] = f"Pick the {items} from the counter and place {'them' if self.num_food > 1 else 'it'} in the cabinet. Then close the cabinet."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        for i in range(self.num_food):
            cfgs.append(
                dict(
                    name=f"food{i}",
                    obj_groups=["fruit", "vegetable", "boxed_food"],
                    graspable=True,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.cab,
                        ),
                        size=(0.30, 0.30),
                        pos=("ref", -1.0),
                        offset=(0.05, 0.0),
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        food_inside_cab = all(
            [OU.obj_inside_of(self, f"food{i}", self.cab) for i in range(self.num_food)]
        )
        cab_closed = True
        door_state = self.cab.get_door_state(env=self)

        for joint_p in door_state.values():
            if joint_p > 0.05:
                cab_closed = False
                break
        return cab_closed and food_inside_cab
