from robocasa.environments.kitchen.kitchen import *


class ColorfulSalsa(Kitchen):

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(1.0, 0.4)))
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = f"Place the avocado, onion, tomato and bell pepper on the cutting board"

        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        

    def _get_obj_cfgs(self):
        cfgs = []        
        cfgs.append(dict(
            name="receptacle",
            obj_groups="cutting_board",
            graspable=False,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    top_size=(1.0, 0.4)
                ),
                size=(1, 0.4),
                pos=(0, -1),
            ),
        ))

        

        cfgs.append(dict(
                name="bell_pepper",
                obj_groups="bell_pepper",
                placement=dict(
                    fixture=self.counter,
                    # sample_region_kwargs=dict(
                    #     top_size=(1.0, 0.4)
                    # ),
                    size=(1, 0.4),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tomato",
                obj_groups="tomato",
                placement=dict(
                    fixture=self.counter,
                    # sample_region_kwargs=dict(
                    #     top_size=(1.0, 0.4)
                    # ),
                    size=(1, 0.4),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="avocado",
                obj_groups="avocado",
                placement=dict(
                    fixture=self.counter,
                    # sample_region_kwargs=dict(
                    #     top_size=(1.0, 0.4)
                    # ),
                    size=(1, 0.4),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="onion",
                obj_groups="onion",
                placement=dict(
                    fixture=self.counter,
                    # sample_region_kwargs=dict(
                    #     top_size=(1.0, 0.4)
                    # ),
                    size=(1, 0.4),
                    pos=(0, -1),
                ),
            )
        )

        

        return cfgs

    def _check_success(self):
        vegetables_on_board = OU.check_obj_in_receptacle(self, "onion", "receptacle")  and OU.check_obj_in_receptacle(self, "avocado", "receptacle") and OU.check_obj_in_receptacle(self, "tomato", "receptacle") and OU.check_obj_in_receptacle(self, "bell_pepper", "receptacle")
        

        return vegetables_on_board and OU.gripper_obj_far(self, "receptacle")