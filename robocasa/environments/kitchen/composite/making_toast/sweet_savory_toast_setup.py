from robocasa.environments.kitchen.kitchen import *


class SweetSavoryToastSetup(Kitchen):
    """
    Sweet Savory Toast Setup: composite task for Making Toast activity.

    Simulates the task of setting up the ingredients for making sweet and savory
    toast.

    Steps:
        Pick the avocado and bread from the counter and place it on the plate.
        Then pick the jam from the cabinet and place it next to the plate.
        Lastly, close the cabinet door.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(1.0, 0.5))
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the avocado and bread from the counter and place them on the plate. "
            "Then pick the jam from the cabinet and place it next to the plate. "
            "Lastly close the cabinet door."
        )
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="avocado",
                obj_groups="avocado",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="plate",
                    size=(1.0, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread",
                object_scale=0.80,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="plate",
                    size=(1.0, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="jam",
                obj_groups="jam",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.3),
                    pos=(0, -1.0),
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
        gripper_obj_far = OU.gripper_obj_far(self, "plate")
        jam_on_counter = self.check_contact(self.objects["jam"], self.counter)
        food_on_plate = OU.check_obj_in_receptacle(
            self, "bread", "plate"
        ) and OU.check_obj_in_receptacle(self, "avocado", "plate")
        cab_closed = self.cab.is_closed(env=self)

        jam_pos = self.sim.data.body_xpos[self.obj_body_id["jam"]]
        plate_pos = self.sim.data.body_xpos[self.obj_body_id["plate"]]
        jam_plate_close = np.linalg.norm(jam_pos - plate_pos) < 0.25

        return (
            gripper_obj_far
            and food_on_plate
            and jam_on_counter
            and jam_plate_close
            and cab_closed
        )
