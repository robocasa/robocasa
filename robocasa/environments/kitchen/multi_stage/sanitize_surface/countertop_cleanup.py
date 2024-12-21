from robocasa.environments.kitchen.kitchen import *


class CountertopCleanup(Kitchen):
    """
    Countertop Cleanup: composite task for Sanitize Surface activity.

    Simulates the task of cleaning the countertop.

    Steps:
        Pick the fruit and vegetable from the counter and place it in the cabinet.
        Then, open the drawer and pick the cleaner and sponge from the drawer and
        place it on the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):

        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET_TOP))
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER, ref=self.cab)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )

        self.init_robot_base_pos = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the fruit and vegetable from the counter and place them in the cabinet. "
            "Then open the drawer and pick the cleaner and sponge from the drawer and place them on the counter."
        )
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """

        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        # objects appear on different sides
        direction = self.rng.choice([1.0, -1.0])

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("spray", "bar_soap", "soap_dispenser"),
                graspable=True,
                placement=dict(
                    fixture=self.drawer,
                    size=(0.3, 0.3),
                    pos=(-1.0 * direction, -0.5),
                    rotation=np.pi / 2,
                    rotation_axis="x",
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="sponge",
                graspable=True,
                placement=dict(
                    fixture=self.drawer,
                    size=(0.3, 0.3),
                    pos=(1.0 * direction, -0.5),
                ),
            )
        ),

        cfgs.append(
            dict(
                name="obj3",
                obj_groups=("fruit"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0.0, -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj4",
                obj_groups=("vegetable"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0.0, -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self) and OU.gripper_obj_far(self, "obj3")
        objs_on_counter = OU.check_obj_fixture_contact(
            self, "obj", self.counter
        ) and OU.check_obj_fixture_contact(self, "obj2", self.counter)
        objs_inside_cab = OU.obj_inside_of(self, "obj3", self.cab) and OU.obj_inside_of(
            self, "obj4", self.cab
        )

        return gripper_obj_far and objs_inside_cab and objs_on_counter
