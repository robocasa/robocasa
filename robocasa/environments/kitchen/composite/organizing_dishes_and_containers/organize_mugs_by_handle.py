from robocasa.environments.kitchen.kitchen import *


class OrganizeMugsByHandle(Kitchen):
    """
    Organize Mugs by Handle Direction: composite task for Organizing Dishes and Containers.

    Simulates the task of organizing mugs in the cabinet such that all handles face the right.

    Steps:
        Pick and place the mug from the counter to the cabinet with the handle facing to the right.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the mug on the counter and place it in the cabinet with the handle facing the right, like the mugs already inside."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"mug{i+1}",
                    obj_groups="mug",
                    placement=dict(
                        fixture=self.cabinet,
                        size=(0.40, 0.20),
                        pos=(-0.9 + (i * 0.3), 0.5),
                        rotation=(np.pi / 2, np.pi / 2),
                    ),
                )
            )
        cfgs.append(
            dict(
                name=f"mug_counter1",
                obj_groups="mug",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(1.0, 0.30),
                    pos=("ref", -0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        mug_orientation = self.sim.data.body_xquat[self.obj_body_id["mug_counter1"]]
        yaw = 2 * np.arctan2(mug_orientation[3], mug_orientation[0]) % (2 * np.pi)

        cabinet_yaw = self.cabinet.rot
        min_yaw = (cabinet_yaw + np.deg2rad(50)) % (2 * np.pi)
        max_yaw = (cabinet_yaw + np.deg2rad(130)) % (2 * np.pi)

        inside = OU.obj_inside_of(self, "mug_counter1", self.cabinet)
        if not inside:
            return False

        if (min_yaw < max_yaw and not (min_yaw <= yaw <= max_yaw)) or (
            min_yaw > max_yaw and not (yaw >= min_yaw or yaw <= max_yaw)
        ):
            return False

        gripper_far = OU.gripper_obj_far(self, obj_name="mug_counter1")
        if gripper_far:
            return True

        return False
