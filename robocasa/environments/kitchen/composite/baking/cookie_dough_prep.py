from robocasa.environments.kitchen.kitchen import *


class CookieDoughPrep(Kitchen):
    """
    Cookie Dough Prep: composite task for Baking Cookies and Cakes activity.
    Simulates the task of preparing cookie dough.
    Steps:
        Pick an egg and butter from the fridge and place it by the bowl for mixing.
    """

    def __init__(
        self, obj_registries=("objaverse", "lightwheel", "aigen"), *args, **kwargs
    ):
        obj_registries = list(obj_registries or [])
        # make sure to use aigen objects to access the butter
        if "aigen" not in obj_registries:
            obj_registries.append("aigen")
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the egg and butter from the fridge and place them by the bowl for mixing."
        return ep_meta

    def _setup_scene(self):
        self.fridge.open_door(self)
        self.cabinet.open_door(self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="egg",
                obj_groups=("egg"),
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.25),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(z_range=(1, 1.5)),
                ),
            )
        )

        cfgs.append(
            dict(
                name="butter",
                obj_groups=("butter_stick"),
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.25),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(z_range=(1, 1.5)),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr1",
                fridgable=True,
                exclude_obj_groups=("egg", "butter_stick"),
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(0, 0),
                    sample_region_kwargs=dict(),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="flour_bag",
                obj_groups="flour_bag",
                placement=dict(fixture=self.cabinet, size=(0.3, 0.3), pos=(0, 0)),
            )
        )
        return cfgs

    def _check_success(self):
        egg_on_counter = OU.check_obj_any_counter_contact(self, "egg")
        butter_on_counter = OU.check_obj_any_counter_contact(self, "butter")

        distance_thresh = 0.25
        bowl_pos_xy = self.sim.data.body_xpos[self.obj_body_id["bowl"]][0:2]
        bowl_radius = np.linalg.norm(self.objects["bowl"].horizontal_radius)

        egg_pos_xy = self.sim.data.body_xpos[self.obj_body_id["egg"]][0:2]
        butter_pos_xy = self.sim.data.body_xpos[self.obj_body_id["butter"]][0:2]

        egg_near_bowl = np.linalg.norm(egg_pos_xy - bowl_pos_xy) < (
            bowl_radius + distance_thresh
        )
        butter_near_bowl = np.linalg.norm(butter_pos_xy - bowl_pos_xy) < (
            bowl_radius + distance_thresh
        )
        gripper_egg_far = OU.gripper_obj_far(self, obj_name="egg", th=0.15)
        gripper_butter_far = OU.gripper_obj_far(self, obj_name="butter", th=0.15)
        return (
            egg_on_counter
            and butter_on_counter
            and egg_near_bowl
            and butter_near_bowl
            and gripper_egg_far
            and gripper_butter_far
        )
