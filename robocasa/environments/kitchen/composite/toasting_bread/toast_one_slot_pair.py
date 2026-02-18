from robocasa.environments.kitchen.kitchen import *


class ToastOneSlotPair(Kitchen):
    """
    Toast One Slot Pair: composite task for Toasting Bread activity.

    Simulates the task of troubleshooting a toaster with a broken slot-pair by moving bread
    to the working slot-pair and toasting both slices together.

    Steps:
        1. Identify which slot-pair is broken and which is working
        2. Pick the bread slice from the broken slot-pair
        3. Place it in the working slot-pair alongside the existing bread slice
        4. Press the lever to toast both slices together
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS
    # double slot-pair styles only
    EXCLUDE_STYLES = [
        1,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        15,
        17,
        20,
        21,
        24,
        27,
        28,
        32,
        35,
        36,
        39,
        43,
        44,
        46,
        47,
        49,
        51,
        53,
        54,
        56,
        58,
        60,
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "refs" in self._ep_meta:
            self.working_slot_pair = self._ep_meta["refs"]["working_slot_pair"]
            self.broken_slot_pair = self._ep_meta["refs"]["broken_slot_pair"]
        else:
            self.working_slot_pair = self.rng.choice(["left", "right"])
            self.broken_slot_pair = (
                "right" if self.working_slot_pair == "left" else "left"
            )

        self.toaster = self.register_fixture_ref(
            "toaster", dict(id=FixtureType.TOASTER)
        )

        self.init_robot_base_ref = self.toaster

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = (
            f"Move the bread slice from the broken {self.broken_slot_pair} slot to the working {self.working_slot_pair} slot-pair "
            f"so both slices can toast together. Then press that lever to start toasting."
        )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["working_slot_pair"] = self.working_slot_pair
        ep_meta["refs"]["broken_slot_pair"] = self.broken_slot_pair
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        broken_slot_idx = 0 if self.broken_slot_pair == "left" else 1
        working_slot_idx = 0 if self.working_slot_pair == "left" else 1

        cfgs.append(
            dict(
                name="bread_broken",
                obj_groups=("sandwich_bread"),
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    sample_region_kwargs=dict(slot_pair=broken_slot_idx),
                    rotation=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bread_working",
                obj_groups=("sandwich_bread"),
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    sample_region_kwargs=dict(slot_pair=working_slot_idx),
                    rotation=(0, 0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        working_slot_idx = 0 if self.working_slot_pair == "left" else 1

        bread_broken_in_working = self.toaster.check_slot_contact(
            self, "bread_broken", slot_pair=working_slot_idx
        )
        bread_working_in_working = self.toaster.check_slot_contact(
            self, "bread_working", slot_pair=working_slot_idx
        )

        toaster_on = False
        if bread_broken_in_working and bread_working_in_working:
            toaster_state = self.toaster.get_state(self, slot_pair=working_slot_idx)
            toaster_on = toaster_state["turned_on"]

        return bread_broken_in_working and bread_working_in_working and toaster_on
