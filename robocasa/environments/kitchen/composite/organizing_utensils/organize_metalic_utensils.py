from robocasa.environments.kitchen.kitchen import *


class OrganizeMetallicUtensils(Kitchen):
    """
    Organize Metallic Utensils: composite task for Organizing Utensils activity.

    Simulates the task of organizing utensils by their material type.
    The task is to take the metallic utensils and put them in the drawer.

    Steps:
        Identify the metallic utensils from the mix on the dining counter.
        Pick up the metallic utensils from the counter and place them in the drawer.
        Ensure that no other types of utensils are moved.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        if "refs" in self._ep_meta:
            self.third_utensil_is_metallic = self._ep_meta["refs"][
                "third_utensil_is_metallic"
            ]
        else:
            self.third_utensil_is_metallic = self.rng.random() > 0.5

        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Pick up the metallic utensils from the counter and place them in the open drawer. "
            f"Make sure not to pick up any wooden utensils."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["third_utensil_is_metallic"] = self.third_utensil_is_metallic
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.drawer.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="wooden_spoon",
                obj_groups="wooden_spoon",
                object_scale=[1, 1, 1.5],
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.0, 0.35),
                    pos=(0, -1.0),
                    rotation=np.pi,
                ),
            )
        )

        cfgs.append(
            dict(
                name="metallic_utensil",
                obj_groups=("spoon", "fork"),
                object_scale=[1, 1, 2.5],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.0, 0.35),
                    pos=(0, -1.0),
                    rotation=np.pi,
                ),
            )
        )

        cfgs.append(
            dict(
                name="metallic_utensil_2",
                obj_groups="knife",
                object_scale=[1, 1, 2],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.0, 0.35),
                    pos=(0, -1.0),
                    rotation=np.pi,
                ),
            )
        )

        if self.third_utensil_is_metallic:
            cfgs.append(
                dict(
                    name="third_metallic_utensil",
                    obj_groups=("spoon", "fork"),
                    object_scale=[1, 1, 2.5],
                    placement=dict(
                        fixture=self.dining_counter,
                        sample_region_kwargs=dict(
                            ref=self.stool,
                        ),
                        size=(1.0, 0.35),
                        pos=(0, -1.0),
                        rotation=np.pi,
                    ),
                )
            )
        else:
            cfgs.append(
                dict(
                    name="second_wooden_spoon",
                    obj_groups="wooden_spoon",
                    object_scale=[1, 1, 1.5],
                    placement=dict(
                        fixture=self.dining_counter,
                        sample_region_kwargs=dict(
                            ref=self.stool,
                        ),
                        size=(1.0, 0.35),
                        pos=(0, -1.0),
                        rotation=np.pi,
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        metals = ["metallic_utensil", "metallic_utensil_2"]
        if self.third_utensil_is_metallic:
            metals.append("third_metallic_utensil")

        for m in metals:
            inside = OU.obj_inside_of(self, m, self.drawer)
            on_counter = OU.check_obj_fixture_contact(self, m, self.dining_counter)
            if not inside or on_counter:
                return False

        woods = ["wooden_spoon"]
        if not self.third_utensil_is_metallic:
            woods.append("second_wooden_spoon")

        for w in woods:
            on_counter = OU.check_obj_fixture_contact(self, w, self.dining_counter)
            inside = OU.obj_inside_of(self, w, self.drawer)
            if not on_counter or inside:
                return False

        for obj in metals + woods:
            far = OU.gripper_obj_far(self, obj)
            if not far:
                return False

        return True
