from robocasa.environments.kitchen.kitchen import *


class OpenBlenderLid(Kitchen):
    """
    Class encapsulating the atomic open blender lid task.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the open blender lid task.
        """
        super()._setup_kitchen_references()
        self.blender = self.get_fixture(FixtureType.BLENDER)
        self.init_robot_base_ref = self.blender

    def get_ep_meta(self):

        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the blender by taking off the lid and placing it on the counter."
        return ep_meta

    def _check_success(self):
        # check lid contact with any counter
        gripper_lid_far = OU.gripper_fxtr_far(
            self, f"{self.blender.blender_lid.name}_main", th=0.15
        )
        if not gripper_lid_far:
            return False
        # check if blender lid is on the counter
        for fixture in self.fixtures.values():
            if isinstance(fixture, Counter):
                if self.check_contact(self.blender.blender_lid, fixture):
                    return True
        return False


class CloseBlenderLid(Kitchen):
    """
    Class encapsulating the atomic close blender lid task.
    """

    # update yaml config to place the blender lid next to the blender instead of on top of it
    _BLENDER_PLACEMENT_UPDATE_DICT = {
        "aux_fixture_config": {
            "has_free_joints": True,
            "placement": "parent_region",
            "auxiliary_placement_args": {"ensure_valid_placement": True},
        }
    }

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

        super()._setup_kitchen_references()
        self.blender = self.get_fixture(FixtureType.BLENDER)
        self.init_robot_base_ref = self.blender

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Close the lid blender by securely placing the lid on top."
        return ep_meta

    def _check_success(self):
        """
        Check if the blender lid is closed.
        """
        return self.blender.get_state()["lid_on_blender"] and OU.gripper_fxtr_far(
            self, f"{self.blender.blender_lid.name}_main", th=0.15
        )


class TurnOnBlender(Kitchen):
    """
    Class encapsulating the atomic turn on blender task.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.blender = self.get_fixture(FixtureType.BLENDER)
        self.init_robot_base_ref = self.blender

    def get_ep_meta(self):

        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Turn on the blender by pressing the power button."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("fruit"),
                object_scale=0.80,
                placement=dict(
                    fixture=self.blender,
                    size=(0.40, 0.40),
                    pos=(0, 0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return self.blender.get_state()["turned_on"]
