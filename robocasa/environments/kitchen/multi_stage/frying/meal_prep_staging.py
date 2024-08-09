from robocasa.environments.kitchen.kitchen import *


class MealPrepStaging(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.3, 0.2))
        )
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name_1 = self.get_obj_lang("vegetable")
        obj_name_2 = self.get_obj_lang("meat")
        ep_meta[
            "lang"
        ] = f"pick place both pans onto different burners. Then, place the {obj_name_1} and the {obj_name_2} on different pans"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):

        cfgs = []

        cfgs.append(
            dict(
                name="pan1",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove, loc="left_right"),
                    size=(0.5, 0.05),
                    pos=("ref", 0.2),
                    offset=(-0.13, 0.0),
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="pan2",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.5, 0.05),
                    pos=("ref", -0.3),
                    offset=(-0.13, 0.0),
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=("vegetable"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove, loc="left_right"),
                    size=(0.4, 0.4),
                    pos=("ref", 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups=("meat"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove, loc="left_right"),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_obj_location_on_stove(self, obj_name, threshold=0.08):

        obj = self.objects[obj_name]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])[0:2]
        obj_on_stove = OU.check_obj_fixture_contact(self, obj_name, self.stove)

        if obj_on_stove:
            for location, site in self.stove.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(
                        self.sim.data.get_site_xpos(site.get("name"))
                    )[0:2]
                    dist = np.linalg.norm(burner_pos - obj_pos)

                    obj_on_site = dist < threshold
                    if obj_on_site:
                        return location

        return None

    def _check_success(self):

        vegetable_on_pan1 = OU.check_obj_in_receptacle(self, "vegetable", "pan1")
        vegetable_on_pan2 = OU.check_obj_in_receptacle(self, "vegetable", "pan2")
        meat_on_pan1 = OU.check_obj_in_receptacle(self, "meat", "pan1")
        meat_on_pan2 = OU.check_obj_in_receptacle(self, "meat", "pan2")

        food_on_pans = (vegetable_on_pan1 and meat_on_pan2) or (
            vegetable_on_pan2 and meat_on_pan1
        )

        pan1_loc = self._check_obj_location_on_stove(obj_name="pan1")
        pan2_loc = self._check_obj_location_on_stove(obj_name="pan2")

        pans_on_stove = pan1_loc != None and pan2_loc != None
        pans_diff = pan1_loc != pan2_loc

        return pans_on_stove and pans_diff and food_on_pans
