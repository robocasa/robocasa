from robocasa.environments.kitchen.kitchen import *


class MeatTransfer(Kitchen):
    """
    Meat Transfer: composite task for Chopping Food activity.

    Simulates the task of transferring meat to a container.

    Steps:
        Retrieve a container (either a pan or a bowl) from the cabinet, then place
        the raw meat into the container to avoid contamination.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.5, 0.5))
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        cont_name = self.get_obj_lang("container")
        ep_meta["lang"] = (
            f"Retrieve the {cont_name} from the cabinet, "
            f"then place the raw meat into the {cont_name} to avoid contamination."
        )
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
                name="container",
                obj_groups=("pan", "bowl", "tupperware"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.5),
                    pos=(0, -1.0),
                    # apply a custom rotation for the pan so that it fits better in the cabinet
                    # (if the handle sticks out it may not fit)
                    rotation=(2 * np.pi / 8, 3 * np.pi / 8),
                ),
            )
        )
        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(0.5, 0.4),
                    pos=(0.0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        return (
            OU.check_obj_fixture_contact(self, "container", self.counter)
            and OU.gripper_obj_far(self, obj_name="meat")
            and OU.check_obj_in_receptacle(self, "meat", "container")
        )
