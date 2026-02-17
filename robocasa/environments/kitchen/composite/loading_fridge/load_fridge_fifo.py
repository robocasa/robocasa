from robocasa.environments.kitchen.kitchen import *


class LoadFridgeFifo(Kitchen):
    """
    Load Fridge FIFO: composite task for Loading Fridge.

    Simulates the task of placing newer meat behind older meat on the same rack
    to maintain a FIFO (First In, First Out) system.

    Steps:
        Place the newer meat from the counter behind the older meat in the fridge
        on the same rack to maintain FIFO order. Close the fridge door when finished.
    """

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
            self.rack_index = self._ep_meta["refs"]["rack_index"]
            self.meat_type = self._ep_meta["refs"]["meat_type"]
        else:
            self.rack_index = -1 if self.rng.random() < 0.5 else -2
            meat_categories = ["steak", "chicken_drumstick", "fish"]
            self.meat_type = self.rng.choice(meat_categories)

        self.init_robot_base_ref = self.counter

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("meat_old")

        ep_meta["lang"] = (
            f"Place the {meat_lang} on the counter behind the older {meat_lang} "
            f"on the same shelf of the fridge to maintain FIFO order. Close the fridge door when finished."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["rack_index"] = self.rack_index
        ep_meta["refs"]["meat_type"] = self.meat_type
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat_old",
                obj_groups=self.meat_type,
                object_scale=[1, 1, 1.2],
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.rack_index,
                    ),
                    size=(0.50, 0.15),
                    pos=(0, -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat_new",
                obj_groups=self.meat_type,
                object_scale=[1, 1, 1.2],
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                        full_depth_region=True,
                    ),
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        meat_old_on_rack = self.fridge.check_rack_contact(
            self, "meat_old", compartment="fridge", rack_index=self.rack_index
        )

        meat_new_on_rack = self.fridge.check_rack_contact(
            self, "meat_new", compartment="fridge", rack_index=self.rack_index
        )

        meat_old_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["meat_old"]])
        meat_new_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["meat_new"]])

        _, meat_old_y = OU.transform_global_to_local(
            meat_old_pos[0], meat_old_pos[1], -self.fridge.rot
        )
        _, meat_new_y = OU.transform_global_to_local(
            meat_new_pos[0], meat_new_pos[1], -self.fridge.rot
        )

        # 0.1 buffer
        fifo_order = meat_new_y > meat_old_y + 0.1
        frige_closed = self.fridge.is_closed(self)

        return meat_old_on_rack and meat_new_on_rack and fifo_order and frige_closed
