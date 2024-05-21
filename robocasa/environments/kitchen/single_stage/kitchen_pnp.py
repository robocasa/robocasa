from robocasa.environments.kitchen.kitchen import *


class PnP(Kitchen):
    def __init__(
        self,
        obj_groups="all",
        exclude_obj_groups=None,
        *args,
        **kwargs
    ):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups

        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        raise NotImplementedError


class PnPCounterToCab(PnP):
    def __init__(self, cab_id=FixtureType.CABINET_TOP, obj_groups="all", *args, **kwargs):        
        self.cab_id = cab_id
        super().__init__(obj_groups=obj_groups, *args, **kwargs)
    
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=self.cab_id)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"pick the {obj_lang} from the counter and place it in the cabinet"
        return ep_meta
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
    
    def _get_obj_cfgs(self):
        cfgs = []        
        cfgs.append(dict(
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
                pos=(0.0, -1.0),
                offset=(0.0, 0.10),
            ),
        ))

        # distractors
        cfgs.append(dict(
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
        ))
        cfgs.append(dict(
            name="distr_cab",
            obj_groups="all",
            placement=dict(
                fixture=self.cab,
                size=(1.0, 0.20),
                pos=(0.0, 1.0),
                offset=(0.0, 0.0),
            ),
        ))

        return cfgs

    def _check_success(self):
        obj_inside_cab = OU.obj_inside_of(self, "obj", self.cab)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_inside_cab and gripper_obj_far


class PnPCabToCounter(PnP):
    def __init__(self, cab_id=FixtureType.CABINET_TOP, obj_groups="all", *args, **kwargs):        
        self.cab_id = cab_id
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=self.cab_id),
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab),
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"pick the {obj_lang} from the cabinet and place it on the counter"
        return ep_meta
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
    
    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True,
            placement=dict(
                fixture=self.cab,
                size=(0.50, 0.20),
                pos=(0, -1.0),
            ),
        ))

        # distractors
        cfgs.append(dict(
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
        ))
        cfgs.append(dict(
            name="distr_cab",
            obj_groups="all",
            placement=dict(
                fixture=self.cab,
                size=(1.0, 0.20),
                pos=(0.0, 1.0),
                offset=(0.0, 0.0),
            ),
        ))

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self)
        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        return obj_on_counter and gripper_obj_far


class PnPCounterToSink(PnP):
    def __init__(self, obj_groups="all", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink", dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"pick the {obj_lang} from the counter and place it in the sink"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True, washable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                ),
                size=(0.30, 0.40),
                pos=("ref", -1.0),
            ),
        ))
        
        # distractors
        cfgs.append(dict(
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
        ))
        cfgs.append(dict(
            name="distr_sink",
            obj_groups="all",
            washable=True,
            placement=dict(
                fixture=self.sink,
                size=(0.25, 0.25),
                pos=(0.0, 1.0),
            ),
        ))

        return cfgs

    def _check_success(self):
        obj_in_sink = OU.obj_inside_of(self, "obj", self.sink)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_in_sink and gripper_obj_far


class PnPSinkToCounter(PnP):
    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink", dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta["lang"] = f"pick the {obj_lang} from the sink and place it on the {cont_lang} located on the counter"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True, washable=True,
            placement=dict(
                fixture=self.sink,
                size=(0.25, 0.25),
                pos=(0.0, 1.0),
            ),
        ))
        cfgs.append(dict(
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
        ))
        
        # distractors
        cfgs.append(dict(
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
        ))

        return cfgs

    def _check_success(self):
        obj_in_recep = OU.check_obj_in_receptacle(self, "obj", "container")
        recep_on_counter = self.check_contact(self.objects["container"], self.counter)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_in_recep and recep_on_counter and gripper_obj_far


class PnPCounterToMicrowave(PnP):
    EXCLUDE_LAYOUTS = [8]
    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)
    
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter", dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_pos = self.microwave
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"pick the {obj_lang} from the counter and place it in the microwave"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True, microwavable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.microwave,
                ),
                size=(0.30, 0.30),
                pos=("ref", -1.0),
                try_to_place_in="container",
            ),
        ))
        cfgs.append(dict(
            name="container",
            obj_groups=("plate"),
            placement=dict(
                fixture=self.microwave,
                size=(0.05, 0.05),
                ensure_object_boundary_in_range=False,
            ),
        ))

        # distractors
        cfgs.append(dict(
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
        ))

        return cfgs

    def _check_success(self):
        obj = self.objects["obj"]
        container = self.objects["container"]

        obj_container_contact = self.check_contact(obj, container)
        container_micro_contact = self.check_contact(container, self.microwave)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_container_contact and container_micro_contact and gripper_obj_far
    

class PnPMicrowaveToCounter(PnP):
    EXCLUDE_LAYOUTS = [8]
    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter", dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_pos = self.microwave
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta["lang"] = f"pick the {obj_lang} from the microwave and place it on {cont_lang} located on the counter"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True, microwavable=True,
            placement=dict(
                fixture=self.microwave,
                size=(0.05, 0.05),
                ensure_object_boundary_in_range=False,
                try_to_place_in="container",
            ),
        ))
        cfgs.append(dict(
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
        ))

        # distractors
        cfgs.append(dict(
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
        ))

        return cfgs

    def _check_success(self):
        obj_container_contact = OU.check_obj_in_receptacle(self, "obj", "container")
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_container_contact and gripper_obj_far


class PnPCounterToStove(PnP):
    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref(
            "stove", dict(id=FixtureType.STOVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta["lang"] = f"pick the {obj_lang} from the plate and place it in the {cont_lang}"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []
        
        cfgs.append(dict(
            name="container",
            obj_groups=("pan"),
            placement=dict(
                fixture=self.stove,
                ensure_object_boundary_in_range=False,
                size=(0.02, 0.02),
                rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
            ),
        ))

        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True, cookable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.stove,
                ),
                size=(0.30, 0.30),
                pos=("ref", -1.0),
                try_to_place_in="container",
            ),
        ))

        return cfgs

    def _check_success(self):
        obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container", th=0.07)
        gripper_obj_far = OU.gripper_obj_far(self)

        return obj_in_container and gripper_obj_far
    

class PnPStoveToCounter(PnP):
    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref(
            "stove", dict(id=FixtureType.STOVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        obj_cont_lang = self.get_obj_lang(obj_name="obj_container")
        cont_lang, preposition = self.get_obj_lang(obj_name="container", get_preposition=True)
        ep_meta["lang"] = f"pick the {obj_lang} from the {obj_cont_lang} and place it {preposition} the {cont_lang}"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []
        
        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True, cookable=True,
            max_size=(0.15, 0.15, None),
            placement=dict(
                fixture=self.stove,
                ensure_object_boundary_in_range=False,
                size=(0.02, 0.02),
                rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                try_to_place_in="pan",
            ),
        ))

        cfgs.append(dict(
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
        ))

        return cfgs

    def _check_success(self):
        obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container", th=0.07)
        gripper_obj_far = OU.gripper_obj_far(self)

        return obj_in_container and gripper_obj_far