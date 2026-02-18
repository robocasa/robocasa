from robocasa.environments.kitchen.kitchen import *


class GatherMarinadeIngredients(Kitchen):
    """
    Gather Marinade Ingredients: composite task for Preparing Marinade activity.

    Simulates the task of collecting typical marinade ingredients into one area.

    Steps:
        Retrieve garlic from the fridge and place them in the mixing bowl.
        Retrieve oil/vinegar bottle and shaker from the cabinet and place them
        next to the mixing bowl on the counter.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_WITH_DOOR)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = (
            f"Retrieve the bottle and the shaker from the cabinet "
            f"and place them next to the mixing bowl on the counter. "
            f"Then add garlic from the fridge to the mixing bowl."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)
        self.fridge.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="mixing_bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="garlic",
                obj_groups="garlic",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.3, 0.2),
                    pos=(0.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_fridge",
                obj_groups="food",
                exclude_obj_groups="garlic",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.3, 0.2),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="oil_or_vinegar_bottle",
                obj_groups="oil_and_vinegar_bottle",
                object_scale=0.9,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.6, 0.2),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="shaker",
                obj_groups=("shaker", "salt_and_pepper_shaker"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.6, 0.2),
                    pos=(0.5, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        garlic_in_bowl = OU.check_obj_in_receptacle(self, "garlic", "mixing_bowl")

        oil_on_counter = OU.check_obj_any_counter_contact(self, "oil_or_vinegar_bottle")
        shaker_on_counter = OU.check_obj_any_counter_contact(self, "shaker")

        bowl_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["mixing_bowl"]])[
            :2
        ]
        oil_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["oil_or_vinegar_bottle"]]
        )[:2]
        cond_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["shaker"]])[:2]

        oil_near_bowl = np.linalg.norm(oil_pos - bowl_pos) <= 0.3
        cond_near_bowl = np.linalg.norm(cond_pos - bowl_pos) <= 0.3

        all_far = all(
            OU.gripper_obj_far(self, obj_name=name)
            for name in (
                "garlic",
                "oil_or_vinegar_bottle",
                "shaker",
            )
        )

        bowl_on_counter = OU.check_obj_fixture_contact(
            self, "mixing_bowl", self.counter
        )

        return (
            garlic_in_bowl
            and oil_on_counter
            and shaker_on_counter
            and oil_near_bowl
            and cond_near_bowl
            and all_far
            and bowl_on_counter
        )
