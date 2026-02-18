from robocasa.environments.kitchen.kitchen import *


class ReturnWashingSupplies(Kitchen):
    """
    Put Away Washing Supplies: composite task for Washing Dishes activity.

    The cleaning supplies should be picked up from the sink and placed back in the cabinet.

    Steps:
        1. Pick up the first cleaning item from the sink.
        2. Place the cleaning item back in the cabinet.
        3. Repeat for the second cleaning item.
        4. Ensure the cabinet doors are closed.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang1 = self.get_obj_lang("supply1")
        obj_lang2 = self.get_obj_lang("supply2")
        obj_lang = f"{obj_lang1} and {obj_lang2}"
        if isinstance(self.cabinet, SingleCabinet):
            num_doors = 1
        elif isinstance(self.cabinet, HingeCabinet):
            num_doors = 2
        else:
            num_doors = 0
        ep_meta[
            "lang"
        ] = f"Pick up the {obj_lang} from the sink, and place them back in the {self.cabinet.nat_lang}"
        if num_doors > 0:
            door_lang = "doors" if num_doors > 1 else "door"
            ep_meta["lang"] += f", and close the cabinet {door_lang}."
        else:
            ep_meta["lang"] += "."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="supply1",
                obj_groups="cleaner",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.4, 0.3),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="supply2",
                obj_groups="cleaner",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.4, 0.3),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Success criteria for the task:
        1. All supplies are inside the cabinet.
        2. The cabinet door is closed.
        """

        supplies_inside_cabinet = all(
            [
                OU.obj_inside_of(self, supply, self.cabinet)
                for supply in ["supply1", "supply2"]
            ]
        )
        cabinet_closed = self.cabinet.is_closed(env=self)

        return supplies_inside_cabinet and cabinet_closed
