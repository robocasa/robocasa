from robocasa.environments.kitchen.kitchen import *


class ManipulateDoor(Kitchen):
    """
    Class encapsulating the atomic manipulate door tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            door manipulation behavior for the task.

        fixture_id (str): The door fixture id to manipulate.
    """

    def __init__(self, fixture_id, behavior="open", *args, **kwargs):
        assert behavior in ["open", "close"]
        self.fixture_id = fixture_id
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the door tasks.
        """
        super()._setup_kitchen_references()
        self.fxtr = self.register_fixture_ref("fxtr", dict(id=self.fixture_id))
        self.init_robot_base_pos = self.fxtr

    def get_ep_meta(self):
        """
        Get the episode metadata for the door tasks.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        if isinstance(self.fxtr, HingeCabinet):
            door_name = "doors"
        else:
            door_name = "door"
        ep_meta["lang"] = f"{self.behavior} the {self.fxtr.nat_lang} {door_name}"
        return ep_meta

    def _setup_scene(self):
        """
        Reset the environment internal state for the door tasks.
        This includes setting the door state based on the behavior.
        """
        if self.behavior == "open":
            self.fxtr.close_door(env=self)
        elif self.behavior == "close":
            self.fxtr.open_door(env=self)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._setup_scene()

    def _check_success(self):
        """
        Check if the door manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        if self.behavior == "open":
            return self.fxtr.is_open(env=self)
        elif self.behavior == "close":
            return self.fxtr.is_closed(env=self)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the door tasks. This includes the object placement configurations.
        Place one object inside the door fixture and 1-4 distractors on the counter.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="door_obj",
                obj_groups="all",
                graspable=True,
                microwavable=(True if isinstance(self.fxtr, Microwave) else None),
                placement=dict(
                    fixture=self.fxtr,
                    size=(0.30, 0.30),
                    pos=(None, -1.0),
                ),
            )
        )

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.fxtr),
                        sample_region_kwargs=dict(
                            ref=self.fxtr,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs


class OpenDoor(ManipulateDoor):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)


class CloseDoor(ManipulateDoor):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)


class OpenCabinet(OpenDoor):
    def __init__(self, fixture_id=FixtureType.CABINET_WITH_DOOR, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class CloseCabinet(CloseDoor):
    def __init__(self, fixture_id=FixtureType.CABINET_WITH_DOOR, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class OpenMicrowave(OpenDoor):
    def __init__(self, fixture_id=FixtureType.MICROWAVE, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class CloseMicrowave(CloseDoor):
    def __init__(self, fixture_id=FixtureType.MICROWAVE, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class OpenFridge(OpenDoor):
    def __init__(self, fixture_id=FixtureType.FRIDGE, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class CloseFridge(CloseDoor):
    def __init__(self, fixture_id=FixtureType.FRIDGE, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class OpenOven(OpenDoor):
    EXCLUDE_LAYOUTS = [0, 2, 4, 5, 9]

    def __init__(self, fixture_id=FixtureType.OVEN, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class CloseOven(CloseDoor):
    EXCLUDE_LAYOUTS = [0, 2, 4, 5, 9]

    def __init__(self, fixture_id=FixtureType.OVEN, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class OpenDishwasher(OpenDoor):
    def __init__(self, fixture_id=FixtureType.DISHWASHER, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)


class CloseDishwasher(CloseDoor):
    def __init__(self, fixture_id=FixtureType.DISHWASHER, *args, **kwargs):
        super().__init__(fixture_id=fixture_id, *args, **kwargs)
