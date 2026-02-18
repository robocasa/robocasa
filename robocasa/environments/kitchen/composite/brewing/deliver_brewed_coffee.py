from robocasa.environments.kitchen.kitchen import *


class DeliverBrewedCoffee(Kitchen):
    """
    Deliver Brewed Coffee: composite task for Brewing activity.

    Simulates the task of delivering brewed coffee to the dining area.

    Steps:
        Pick the brewed coffee (mug) from the coffee machine and place it next to
        the breakfast plate on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine", dict(id=FixtureType.COFFEE_MACHINE)
        )
        self.init_robot_base_ref = self.coffee_machine

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the brewed coffee from the coffee machine and place it next to "
            "the breakfast plate on the dining counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "brewed_coffee", [0.65, 0.45, 0.25, 0.8])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="brewed_coffee",
                obj_groups="mug",
                placement=dict(
                    fixture=self.coffee_machine,
                    ensure_object_boundary_in_range=False,
                    margin=0.0,
                    ensure_valid_placement=False,
                    rotation=(np.pi / 8, np.pi / 4),
                ),
            )
        )

        cfgs.append(
            dict(
                name="breakfast_plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="breakfast_item1",
                obj_groups=("bread_food", "pancake", "waffle"),
                exclude_obj_groups=("hotdog_bun"),
                placement=dict(
                    object="breakfast_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="breakfast_item2",
                obj_groups=("fruit", "cheese"),
                placement=dict(
                    object="breakfast_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        coffee_on_counter = OU.check_obj_fixture_contact(
            self, "brewed_coffee", self.dining_counter
        )
        plate_on_counter = OU.check_obj_fixture_contact(
            self, "breakfast_plate", self.dining_counter
        )

        if coffee_on_counter and plate_on_counter:
            coffee_pos = self.sim.data.body_xpos[self.obj_body_id["brewed_coffee"]][:2]
            plate_pos = self.sim.data.body_xpos[self.obj_body_id["breakfast_plate"]][:2]
            distance = np.linalg.norm(coffee_pos - plate_pos)
            coffee_close_to_plate = distance <= 0.3
        else:
            coffee_close_to_plate = False

        gripper_far = OU.gripper_obj_far(self, obj_name="brewed_coffee")

        return (
            coffee_on_counter
            and plate_on_counter
            and coffee_close_to_plate
            and gripper_far
        )
