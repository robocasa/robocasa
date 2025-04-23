from robocasa.environments.kitchen.kitchen import *


class PrewashFoodAssembly(Kitchen):
    """
    Prewash Food Assembly: composite task for Washing Fruits And Vegetables activity.

    Simulates the process of assembling fruits and vegetables in a bowl and
    prewashing them.

    Steps:
        Pick the fruit/vegetable from the cabinet and place it in the bowl. Then
        pick the bowl and place it in the sink. Then turn on the sink facuet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which food is picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter_cab = self.register_fixture_ref(
            "counter_cab", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.counter_sink = self.register_fixture_ref(
            "counter_sink", dict(id=FixtureType.COUNTER, ref=self.sink)
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_name = self.get_obj_lang("food")
        ep_meta["lang"] = (
            f"Pick the {food_name} from the cabinet and place it in the bowl. "
            "Then pick the bowl and place it in the sink. Then turn on the sink facuet."
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
                name="food",
                obj_groups=["vegetable", "fruit"],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter_cab,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="bowl")
        food_in_bowl = OU.check_obj_in_receptacle(self, "food", "bowl")
        bowl_in_sink = OU.obj_inside_of(self, "bowl", self.sink)
        handle_state = self.sink.get_handle_state(env=self)
        water_on = handle_state["water_on"]
        return gripper_obj_far and food_in_bowl and bowl_in_sink and water_on
