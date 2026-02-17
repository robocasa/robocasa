from robocasa.environments.kitchen.kitchen import *


class PickPlace(Kitchen):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """

    def __init__(self, obj_groups="all", exclude_obj_groups=None, *args, **kwargs):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups

        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        raise NotImplementedError


class PickPlaceCounterToCabinet(PickPlace):
    """
    Class encapsulating the atomic counter to cabinet pick and place task

    Args:
        cab_id (str): The cabinet fixture id to place the object.

        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, cab_id=FixtureType.CABINET, obj_groups="all", *args, **kwargs):

        self.cab_id = cab_id
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to cabinet pick and place task:
        The cabinet to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to cabinet pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the counter and place it in the cabinet."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to cabinet pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the back area of the cabinet.

        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(None, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to cabinet pick and place task is successful.
        Checks if the object is inside the cabinet and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_inside_cab = OU.obj_inside_of(self, "obj", self.cab)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_inside_cab and gripper_obj_far


class PickPlaceCabinetToCounter(PickPlace):
    """
    Class encapsulating the atomic cabinet to counter pick and place task

    Args:
        cab_id (str): The cabinet fixture id to pick the object from.

        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, cab_id=FixtureType.CABINET, obj_groups="all", *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the cabinet to counter pick and place task:
        The cabinet to pick object from and the counter to place it on
        """
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=self.cab_id),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.cab),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        """
        Get the episode metadata for the cabinet to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the cabinet and place it on the counter."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the cabinet to counter pick and place task.
        Puts the target object in the front area of the cabinet. Puts a distractor object on the counter
        and the back area of the cabinet.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the cabinet to counter pick and place task is successful.
        Checks if the object is on the counter and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        gripper_obj_far = OU.gripper_obj_far(self)
        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        return obj_on_counter and gripper_obj_far


class PickPlaceCounterToSink(PickPlace):
    """
    Class encapsulating the atomic counter to sink pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="all", *args, **kwargs):

        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to sink pick and place task:
        The sink to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to sink pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the counter and place it in the sink."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to sink pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the sink.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to sink pick and place task is successful.
        Checks if the object is inside the sink and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_sink = OU.obj_inside_of(self, "obj", self.sink, partial_check=True)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_in_sink and gripper_obj_far


class PickPlaceSinkToCounter(PickPlace):
    """
    Class encapsulating the atomic sink to counter pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="food", *args, **kwargs):

        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the sink to counter pick and place task:
        The sink to pick object from and the counter to place it on
        """
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the sink to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the sink and place it on the {cont_lang} located on the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the sink to counter pick and place task.
        Puts the target object in the sink. Puts a distractor object on the counter
        and places a container on the counter for the target object to be placed on.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups="container",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.35, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the sink to counter pick and place task is successful.
        Checks if the object is in the container, the container on the counter, and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_recep = OU.check_obj_in_receptacle(self, "obj", "container")
        recep_on_counter = self.check_contact(self.objects["container"], self.counter)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_in_recep and recep_on_counter and gripper_obj_far


class PickPlaceCounterToMicrowave(PickPlace):
    """
    Class encapsulating the atomic counter to microwave pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    # exclude layout 9 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [9]

    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to microwave pick and place task:
        The microwave to place object on, the counter to initialize it/the container on, and a distractor counter
        """
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.microwave.open_door(env=self)

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to microwave pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the counter and place it in the microwave."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to microwave pick and place task.
        Puts the target object in a container on the counter. Puts a distractor object on the distractor
        counter and places another container in the microwave.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.distr_counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to microwave pick and place task is successful.
        Checks if the object is inside the microwave and on the container and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj = self.objects["obj"]
        container = self.objects["container"]

        obj_container_contact = self.check_contact(obj, container)
        container_micro_contact = self.check_contact(container, self.microwave)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_container_contact and container_micro_contact and gripper_obj_far


class PickPlaceMicrowaveToCounter(PickPlace):
    """
    Class encapsulating the atomic microwave to counter pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    # exclude layout 9 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [9]

    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the microwave to counter pick and place task:
        The microwave to pick object from, the counter to place it on, and a distractor counter
        """
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.microwave.open_door(env=self)

    def get_ep_meta(self):
        """
        Get the episode metadata for the microwave to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the microwave and place it on {cont_lang} located on the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the microwave to counter pick and place task.
        Puts the target object in a container in the microwave. Puts a distractor object on the distractor
        counter and places another container on the counter."""
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("container"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.distr_counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the microwave to counter pick and place task is successful.
        Checks if the object is inside the container and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_container_contact = OU.check_obj_in_receptacle(self, "obj", "container")
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_container_contact and gripper_obj_far


class PickPlaceCounterToOven(PickPlace):
    """
    Class encapsulating the counter to oven pick and place atomic task
    """

    EXCLUDE_LAYOUTS = Kitchen.OVEN_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.oven)
        )
        if "rack_level" in self._ep_meta:
            self.rack_level = self._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        if self.oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Place the {obj_lang} on the {rack_pos} rack of the oven."
        else:
            ep_meta["lang"] = f"Place the {obj_lang} on the rack of the oven."
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.oven.open_door(self)
        self.oven.slide_rack(self, rack_level=self.rack_level)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("oven_ready"),
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.oven,
                    ),
                    size=(0.45, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        cfgs.append(
            dict(
                name="oven_tray",
                obj_groups=("oven_tray"),
                placement=dict(
                    fixture=self.oven,
                    sample_region_kwargs=dict(
                        rack_level=self.rack_level,
                    ),
                    size=(1.0, 0.45),
                    pos=(0, -1.0),
                    rotation=(-0.1, 0.1),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        obj_container_contact = OU.check_obj_in_receptacle(self, "obj", "oven_tray")
        on_rack = self.oven.check_rack_contact(
            self, "oven_tray", rack_level=self.rack_level
        )
        gripper_far = OU.gripper_obj_far(self, "obj")
        return on_rack and obj_container_contact and gripper_far


class PickPlaceCounterToStove(PickPlace):
    """
    Class encapsulating the atomic counter to stove pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to stove pick and place task:
        The stove to place object on and the counter to initialize it/container on
        """
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to stove pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the plate and place it in the {cont_lang}."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to stove pick and place task.
        Puts the target object in a container on the counter and places pan on the stove.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="container",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to stove pick and place task is successful.
        Checks if the object is on the pan and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container", th=0.07)
        gripper_obj_far = OU.gripper_obj_far(self)

        return obj_in_container and gripper_obj_far


class PickPlaceStoveToCounter(PickPlace):
    """
    Class encapsulating the atomic stove to counter pick and place task
    """

    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the stove to counter pick and place task:
        The counter to place object/container on and the stove to initialize it/the pan on
        """
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        """
        Get the episode metadata for the stove to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        obj_cont_lang = self.get_obj_lang(obj_name="obj_container")
        cont_lang, preposition = self.get_obj_lang(
            obj_name="container", get_preposition=True
        )
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the {obj_cont_lang} and place it {preposition} the {cont_lang}."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the stove to counter pick and place task.
        Puts the target object in a pan on the stove and places a container on the counter.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                cookable=True,
                max_size=(0.15, 0.15, None),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    try_to_place_in="pan",
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate", "bowl"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the stove to counter pick and place task is successful.
        Checks if the object is inside the container on the counter and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container", th=0.07)
        gripper_obj_far = OU.gripper_obj_far(self)

        return obj_in_container and gripper_obj_far


class PickPlaceToasterToCounter(PickPlace):
    """
    Class encapsulating the toaster to counter pick and place atomic task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the toaster to plate task
        """
        super()._setup_kitchen_references()
        self.toaster = self.get_fixture(FixtureType.TOASTER)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster)
        )
        self.init_robot_base_ref = self.toaster

    def get_ep_meta(self):
        """
        Get the episode metadata for the toaster to plate task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the toasted item on a plate."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the toaster to plate task.
        Places a toasted item in the toaster and a plate on the counter.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("sandwich_bread",),
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    rotation=(0, 0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster,
                    ),
                    size=(0.80, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        """
        Check if the toaster to plate task is successful.
        Checks if the object is on the plate and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_on_plate = OU.check_obj_in_receptacle(self, "obj", "plate")
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_on_plate and gripper_obj_far


class PickPlaceCounterToToasterOven(PickPlace):
    """
    Class encapsulating the counter to toaster oven pick and place atomic task
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster_oven)
        )
        if "rack_level" in self._ep_meta:
            self.rack_level = self._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        receptacle_type = "rack" if "rack" in self.chosen_toaster_receptacle else "tray"
        if self.toaster_oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Place the {obj_lang} on the {rack_pos} {receptacle_type} of the toaster oven."
        else:
            ep_meta[
                "lang"
            ] = f"Place the {obj_lang} on the {receptacle_type} of the toaster oven."
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
            self, rack_level=self.rack_level
        )

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("bread_food"),
                graspable=True,
                object_scale=0.80,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.45, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        return cfgs

    def _check_success(self):
        on_rack = self.toaster_oven.check_rack_contact(
            self, "obj", rack_level=self.rack_level
        )
        gripper_far = OU.gripper_obj_far(self, "obj")
        return on_rack and gripper_far


class PickPlaceToasterOvenToCounter(PickPlace):
    """
    Class encapsulating the toaster oven to counter pick and place atomic task
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster_oven)
        )
        if "rack_level" in self._ep_meta:
            self.rack_level = self._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        receptacle_type = "rack" if "rack" in self.chosen_toaster_receptacle else "tray"
        obj_lang = self.get_obj_lang()
        if self.toaster_oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} from the {rack_pos} {receptacle_type} and place it on the plate on the counter."
        else:
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} from the {receptacle_type} and place it on the plate on the counter."
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
            self, rack_level=self.rack_level
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("bread_food"),
                graspable=True,
                object_scale=0.80,
                placement=dict(
                    fixture=self.toaster_oven,
                    sample_region_kwargs=dict(
                        rack_level=self.rack_level,
                    ),
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj_in_recep = OU.check_obj_in_receptacle(self, "obj", "container")
        recep_on_counter = self.check_contact(self.objects["container"], self.counter)
        return obj_in_recep and recep_on_counter and OU.gripper_obj_far(self, "obj")


class PickPlaceCounterToStandMixer(PickPlace):
    """
    Class encapsulating the task of placing food items in the stand mixer bowl.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["stand_mixer"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stand_mixer = self.get_fixture(FixtureType.STAND_MIXER)
        self.counter = self.get_fixture(
            FixtureType.COUNTER_NON_CORNER, ref=self.stand_mixer
        )

        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"Place the {obj_lang} in the stand mixer bowl."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stand_mixer.set_head_pos(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("cheese", "bread", "cake"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stand_mixer,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        """
        Check if the food item is inside the stand mixer bowl.

        Returns:
            bool: True if the food item is inside the bowl, False otherwise.
        """
        return self.stand_mixer.check_item_in_bowl(self, "obj") and OU.gripper_obj_far(
            self, "obj"
        )


class PickPlaceFridgeShelfToDrawer(PickPlace):
    """
    Class encapsulating the atomic task of moving an item from a fridge shelf to the fridge drawer.
    """

    # No drawer in side-by-side fridge in these styles
    EXCLUDE_STYLES = [23, 24, 25, 27, 28, 37, 38, 40, 44, 47, 56]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge
        if "refs" in self._ep_meta:
            self.rack_index = self._ep_meta["refs"]["rack_index"]
        else:
            self.rack_index = -1 if self.rng.random() < 0.5 else -2

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self, min=1.0, max=1.0)
        self.fridge.open_door(
            self, min=0.8, max=1.0, reg_type="drawer", drawer_rack_index=-1
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the fridge shelf and place it in the fridge drawer."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["rack_index"] = self.rack_index
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("vegetable", "fruit"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        reg_type=("shelf"),
                        rack_index=self.rack_index,
                    ),
                    size=(0.40, 0.15),
                    pos=(0.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=("vegetable", "fruit"),
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        reg_type=("shelf"),
                    ),
                    size=(0.40, 0.15),
                    pos=(0.0, -0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        obj_in_drawer = self.fridge.check_rack_contact(
            self, "obj", compartment="fridge", reg_type=("drawer"), rack_index=-1
        )
        return obj_in_drawer and OU.gripper_obj_far(self, "obj", th=0.15)


class PickPlaceFridgeDrawerToShelf(PickPlace):
    """
    Class encapsulating the atomic task of moving an item from the fridge drawer to a fridge shelf.
    """

    # No drawer in side-by-side fridge in these layouts or hard to reach
    EXCLUDE_LAYOUTS = [2, 7, 9, 18, 21, 24, 29, 32, 38, 41, 43, 44, 45, 47, 57]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self, min=1.0, max=1.0)
        self.fridge.open_door(
            self, min=0.8, max=0.8, reg_type="drawer", drawer_rack_index=-1
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the fridge drawer and place it on a fridge shelf."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("vegetable", "fruit"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        reg_type=("drawer"),
                        rack_index=-1,
                    ),
                    size=(0.4, 0.2),
                    pos=(0.0, 0.25),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=("vegetable", "fruit"),
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        reg_type=("shelf"),
                    ),
                    size=(0.4, 0.2),
                    pos=(0.0, 0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        obj_on_rack = self.fridge.check_rack_contact(
            self, "obj", compartment="fridge", reg_type=("shelf")
        )
        return obj_on_rack and OU.gripper_obj_far(self, "obj", th=0.15)


class PickPlaceCounterToDrawer(PickPlace):
    """
    Class encapsulating the atomic task of moving an item from the counter into the drawer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )
        self.init_robot_base_ref = self.drawer

    def _setup_scene(self):
        super()._setup_scene()
        self.drawer.open_door(self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the counter and place it in the drawer."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("tool", "utensil"),
                exclude_obj_groups=("reamer", "strainer", "cheese_grater"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.60, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=("tool", "utensil"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.60, 0.30),
                    pos=("ref", -0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        in_drawer = OU.obj_inside_of(
            self, "obj", self.drawer
        ) and not OU.check_obj_any_counter_contact(self, "obj")
        return in_drawer and OU.gripper_obj_far(self, "obj")


class PickPlaceDrawerToCounter(PickPlace):
    """
    Class encapsulating the atomic task of moving an item from the drawer to the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )
        self.init_robot_base_ref = self.drawer

    def _setup_scene(self):
        super()._setup_scene()
        self.drawer.open_door(self, min=1.0, max=1.0)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the drawer and place it on the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("tool", "utensil"),
                exclude_obj_groups=("reamer", "strainer", "cheese_grater"),
                graspable=True,
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.25),
                    pos=(0, -0.25),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=("tool", "utensil"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.5, 0.30),
                    pos=("ref", -0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        on_counter = OU.check_obj_any_counter_contact(self, "obj")
        return on_counter and OU.gripper_obj_far(self, "obj")


class PickPlaceCounterToBlender(PickPlace):
    """
    Class encapsulating the atomic counter to blender pick and place task.
    """

    # update yaml to disable the lid since we are placing object in blender
    _BLENDER_PLACEMENT_UPDATE_DICT = {"aux_fixture_config": {"enable": False}}

    def __init__(
        self, enable_fixtures=None, update_fxtr_cfg_dict=None, *args, **kwargs
    ):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]

        update_fxtr_cfg_dict = update_fxtr_cfg_dict or {}
        update_fxtr_cfg_dict["blender"] = self._BLENDER_PLACEMENT_UPDATE_DICT
        super().__init__(
            enable_fixtures=enable_fixtures,
            update_fxtr_cfg_dict=update_fxtr_cfg_dict,
            *args,
            **kwargs,
        )

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to blender pick and place task:
        The blender to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()
        self.blender = self.get_fixture(FixtureType.BLENDER)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.blender)
        )
        self.init_robot_base_ref = self.blender

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to blender pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the counter and place it in the blender."
        return ep_meta

    def _get_obj_cfgs(self):

        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("fruit"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.blender,
                        loc="left_right",
                    ),
                    size=(0.50, 0.25),
                    pos=("ref", -1.0),
                ),
                object_scale=0.80,
            )
        )

        # distractor
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                exclude_obj_groups=("fruit", "vegetable"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.blender,
                    ),
                    size=(0.5, 0.30),
                    pos=("ref", -0.50),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        """
        Check if the counter to blender pick and place task is successful.
        Checks if the object is inside the blender and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_blender = OU.obj_inside_of(self, "obj", self.blender, th=0.01)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_in_blender and gripper_obj_far


# Additional PickPlace tasks
class CheesyBread(Kitchen):
    """
    Simulates the task of making cheesy bread.

    Steps:
        Start with a slice of bread already on a plate and a wedge of cheese on the
        counter. Pick up the wedge of cheese and place it on the slice of bread to
        prepare a simple cheese on bread dish.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread_flat",
                object_scale=1.5,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.7),
                    pos=(0, -1.0),
                    try_to_place_in="cutting_board",
                ),
            )
        )
        cfgs.append(
            dict(
                name="cheese",
                obj_groups="cheese",
                init_robot_here=True,
                placement=dict(
                    ref_obj="bread_container",
                    fixture=self.counter,
                    size=(1.0, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        # Distractor on the counter
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(fixture=self.counter, size=(1.0, 0.20), pos=(0, 1.0)),
            )
        )
        return cfgs

    def _check_success(self):
        # Bread is still on the cutting board, and cheese is on top
        return (
            OU.check_obj_in_receptacle(self, "bread", "bread_container")
            and OU.gripper_obj_far(self, obj_name="cheese")
            and self.check_contact(self.objects["cheese"], self.objects["bread"])
        )


class MakeIcedCoffee(Kitchen):
    """
    Make Iced Coffee: Add a single ice cube to a glass of coffee.

    Simulates picking up one ice cube and placing it into a glass cup that represents coffee.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up an ice cube and place it in the glass of coffee."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "cup", [0.65, 0.45, 0.25, 0.8])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ice_bowl",
                obj_groups="bowl",
                object_scale=[1, 1, 0.75],
                init_robot_here=True,
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
                name="ice_cube1",
                obj_groups="ice_cube",
                object_scale=0.8,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice_cube2",
                obj_groups="ice_cube",
                object_scale=0.8,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cup",
                obj_groups="glass_cup",
                object_scale=[1.25, 1.25, 1],
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="ice_bowl",
                    size=(0.5, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        ice1_in_cup = OU.check_obj_in_receptacle(self, "ice_cube1", "cup", th=0.5)
        ice2_in_cup = OU.check_obj_in_receptacle(self, "ice_cube2", "cup", th=0.5)
        ice_in_cup = ice1_in_cup or ice2_in_cup

        gripper_far_from_ice1 = OU.gripper_obj_far(self, "ice_cube1", th=0.15)
        gripper_far_from_ice2 = OU.gripper_obj_far(self, "ice_cube2", th=0.15)
        gripper_far = gripper_far_from_ice1 and gripper_far_from_ice2

        return ice_in_cup and gripper_far


class PackDessert(Kitchen):
    """
    Simulates the task of adding dessert to a tupperware that already contains a cooked item.

    Steps:
        1. Take the dessert item and add it to a tupperware that contains a cooked item.
        2. Ensure both items are properly packed in the tupperware.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        sweets_lang = self.get_obj_lang("dessert")
        cooked_lang = self.get_obj_lang("cooked_food")
        ep_meta[
            "lang"
        ] = f"Add the {sweets_lang} to the tupperware that contains the {cooked_lang}."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cooked_food",
                obj_groups="cooked_food",
                init_robot_here=True,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(0, -1.0),
                    try_to_place_in="tupperware",
                    try_to_place_in_kwargs=dict(
                        object_scale=(2.5, 2.5, 1.25),
                    ),
                    rotation=(0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert",
                obj_groups="sweets",
                exclude_obj_groups="sugar_cube",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cooked_food_container",
                    size=(0.5, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return (
            OU.check_obj_in_receptacle(self, "cooked_food", "cooked_food_container")
            and OU.check_obj_in_receptacle(self, "dessert", "cooked_food_container")
            and OU.gripper_obj_far(self, "dessert")
        )
