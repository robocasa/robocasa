from robocasa.environments.kitchen.kitchen import *


class ArrangeTea(Kitchen):
    """
    Arrange Tea: composite task for Brewing activity.

    Simulates the task of arranging tea.

    Steps:
        Take the kettle from the counter and place it on the tray.
        Take the mug from the cabinet and place it on the tray.
        Close the cabinet doors.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # use a double door cabinet so that area below is large enough to initialize all the objects
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        # set the size argument to sample a large enough counter region
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.cab

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick the kettle from the counter and place it on the tray. "
                "Then pick the mug from the cabinet and place it on the tray. "
                "Then close the cabinet doors."
            )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("mug"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("kettle"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                    offset=(0.1, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("tray"),
                placement=dict(
                    fixture=self.counter,
                    size=(0.7, 0.7),
                    pos=("ref", -0.6),
                    offset=(-0.1, 0.0),
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj1_container_contact = OU.check_obj_in_receptacle(self, "obj", "container")
        obj2_container_contact = OU.check_obj_in_receptacle(self, "obj2", "container")
        cab_closed = self.cab.is_closed(env=self)
        gripper_obj_far = OU.gripper_obj_far(
            self
        )  # no need to check all gripper objs far bc all objs in the same place

        return (
            obj1_container_contact
            and obj2_container_contact
            and gripper_obj_far
            and cab_closed
        )
