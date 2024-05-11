from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.objects.fixtures import *


class DefrostByCategory(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "There is a mixed pile of frozen fruits and " \
            "vegetables on the counter. Locate all the frozen vegetables and " \
            "place the items in a bowl on the counter. Take all the frozen " \
            "fruits and defrost them in a running sink."
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []

        # Place the four objects (two fruits, two vegetables)
        placements = list()
        # Making the four regions separate - might help with
        # initialization speed
        for i in range(4):
            placements.append(dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                    top_size=(0.5, 0.5)
                ),
                size=(0.3, 0.4),
                pos=("ref", -1)
            ))
        random.shuffle(placements)

        for i in range(4):
            cfgs.append(dict(
                name="obj" + str(i),
                obj_groups="fruit" if i <= 1 else "vegetable",
                graspable=True,
                placement=placements[i]
            ))

        # Bowl to place the vegetables in
        cfgs.append(dict(
            name="container",
            obj_groups="bowl",
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                    top_size=(0.5, 0.5)
                ),
                size=(0.3, 0.4),
                pos=("ref", -1)
            )
        ))
        return cfgs

    def _check_success(self):
        fruits_in_sink = OU.obj_inside_of(self, "obj0", self.sink) and \
            OU.obj_inside_of(self, "obj1", self.sink)
        vegetables_in_bowl = OU.check_obj_in_receptacle(self, "obj2", "container") and \
            OU.check_obj_in_receptacle(self, "obj3", "container")

        gripper_obj_far = True
        for i in range(4):
            gripper_obj_far = gripper_obj_far and OU.gripper_obj_far(self, obj_name="obj" + str(i))
        

        return fruits_in_sink and vegetables_in_bowl and gripper_obj_far