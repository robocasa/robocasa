from robocasa.environments.kitchen.kitchen import *


class LoadPreparedFood(Kitchen):
    """
    Load Prepared Food: composite task for Loading Fridge.

    Simulates the task of placing prepared food (tupperware with meat and vegetables)
    in the fridge while managing space on the shelves.

    Steps:
        Place the tupperware containing meat and vegetables on the top shelf or
        second highest shelf (randomly decided). The task includes existing items
        on both shelves to create a space management challenge.
    """

    # certain side-by-side fridges only have 1 shelf
    EXCLUDE_STYLES = [11, 15, 18, 22, 34, 45, 49, 52, 53, 54]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge, full_depth_region=True),
        )

        if "refs" in self._ep_meta:
            self.target_rack_index = self._ep_meta["refs"]["target_rack_index"]
        else:
            self.target_rack_index = -1 if self.rng.random() < 0.5 else -2

        self.init_robot_base_ref = self.counter

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("meat_in_tupperware")
        vegetable_lang = self.get_obj_lang("veg_in_tupperware")

        rack_pos = (
            "top shelf" if self.target_rack_index == -1 else "second highest shelf"
        )

        ep_meta["lang"] = (
            f"Place the tupperware on the {rack_pos} of the fridge. Make space for the tupperware "
            f"among the existing items by rearranging them on the shelves if needed. Close the fridge door when finished."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["target_rack_index"] = self.target_rack_index
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="tupperware",
                obj_groups="tupperware",
                object_scale=[1.3, 1.3, 1.2],
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                        full_depth_region=True,
                    ),
                    size=(0.60, 0.50),
                    pos=(0, -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat_in_tupperware",
                obj_groups="meat",
                placement=dict(
                    object="tupperware",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="veg_in_tupperware",
                obj_groups="vegetable",
                placement=dict(
                    object="tupperware",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="existing_item1_top",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-1,
                    ),
                    size=(0.5, 0.25),
                    pos=(-0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="existing_item2_top",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-1,
                    ),
                    size=(0.5, 0.25),
                    pos=(0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="existing_item1_second",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-2,
                    ),
                    size=(0.5, 0.25),
                    pos=(-0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="existing_item2_second",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-2,
                    ),
                    size=(0.5, 0.25),
                    pos=(0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="target_rack_third_item",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.5, 0.4),
                    pos=(0, -0.75),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        tupperware_on_target_rack = self.fridge.check_rack_contact(
            self, "tupperware", compartment="fridge", rack_index=self.target_rack_index
        )

        meat_in_tup = OU.check_obj_in_receptacle(
            self, "meat_in_tupperware", "tupperware"
        )
        veg_in_tup = OU.check_obj_in_receptacle(self, "veg_in_tupperware", "tupperware")

        if not (meat_in_tup or veg_in_tup):
            contents_ok = False
        elif meat_in_tup and veg_in_tup:
            contents_ok = True
        else:
            contents_ok = self.check_contact(
                self.objects["meat_in_tupperware"], self.objects["veg_in_tupperware"]
            )

        existing_items_on_rack = all(
            self.fridge.check_rack_contact(self, name, compartment="fridge")
            for name in [
                "existing_item1_top",
                "existing_item2_top",
                "existing_item1_second",
                "existing_item2_second",
                "target_rack_third_item",
            ]
        )

        fridge_closed = self.fridge.is_closed(self)

        return (
            tupperware_on_target_rack
            and contents_ok
            and existing_items_on_rack
            and fridge_closed
        )
