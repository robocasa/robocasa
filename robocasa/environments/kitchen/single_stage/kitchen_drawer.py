from robocasa.environments.kitchen.kitchen import *


class ManipulateDrawer(Kitchen):
    def __init__(self, behavior="open", drawer_id=FixtureType.TOP_DRAWER, *args, **kwargs):
        self.robot_side = ""
        self.drawer_id = drawer_id
        assert behavior in ["open", "close"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        robot_model = self.robots[0].robot_model
        x_ofs = (self.drawer.width/2) + 0.3
        inits = []
        robot_base_pos_left, robot_base_ori_left = self.compute_robot_base_placement_pose(ref_fixture=self.drawer, offset=(-x_ofs, -0.23))
        test_pos_left, _ = self.compute_robot_base_placement_pose(ref_fixture=self.drawer, offset=(-x_ofs - 0.3, -0.23)) #when checking for contact want there to be space for the robot itself
    
        if not self.check_fxtr_contact(test_pos_left) and not self.check_sidewall_contact(test_pos_left):
            inits.append((robot_base_pos_left, robot_base_ori_left, "right")) #drawer is to the right of the robot

        robot_base_pos_right, robot_base_ori_right = self.compute_robot_base_placement_pose(ref_fixture=self.drawer, offset=(x_ofs, -0.23))
        test_pos_right, _= self.compute_robot_base_placement_pose(ref_fixture=self.drawer, offset=(x_ofs + 0.3, -0.23))

        if not self.check_fxtr_contact(test_pos_right) and not self.check_sidewall_contact(test_pos_right):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left")) 
        
        assert len(inits) > 0
        robot_base_pos, robot_base_ori, side = random.sample(inits, 1)[0]
        self.drawer_side = side
        robot_model.set_base_xpos(robot_base_pos)
        robot_model.set_base_ori(robot_base_ori)
    
    def _reset_internal(self):    
        if self.behavior == "open":
            self.drawer.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.drawer.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        #set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened    
        super()._reset_internal()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=self.drawer_id)
        )
        self.init_robot_base_pos = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta() 
        ep_meta["lang"] = f"{self.behavior} the {self.drawer_side} drawer"
        return ep_meta
    
    def check_fxtr_contact(self, pos):
        fxtrs = [
            fxtr for fxtr in self.fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, SingleCabinet)
            or isinstance(fxtr, HingeCabinet)
        ]

        for fxtr in fxtrs:
            # get bounds of fixture
            if point_in_fixture(point=pos, fixture=fxtr, only_2d=True):
                return True
        return False
    
    def check_sidewall_contact(self, pos):
        walls = [fxtr for (name, fxtr) in self.fixtures.items() if isinstance(fxtr, Wall)]
        for wall in walls:
            if wall.wall_side == "right" and pos[0] > wall.pos[0]:
                return True
            #terrible hack change later we only want to check if its outside the main left wall not a secondary left wall
            if wall.wall_side == "left" and "2" not in wall.name and pos[0]  < wall.pos[0]:
                return True
            if wall.wall_side == "back" and pos[1] > wall.pos[1]:
                return True
        return False
    
    def _check_success(self):
        door_state = self.drawer.get_door_state(env=self)
        
        success = True
        for joint_p in door_state.values():
            if self.behavior == "open":
                if joint_p < 0.95:
                    success = False
                    break
            elif self.behavior == "close":
                if joint_p > 0.05:
                    success = False
                    break

        return success
    

class OpenDrawer(ManipulateDrawer):
    def __init__(self, *args, **kwargs):        
        super().__init__(behavior="open", *args, **kwargs)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="drawer_obj",
            obj_groups="all",
            graspable=True,
            max_size=(None, None, 0.10),
            placement=dict(
                fixture=self.drawer,
                size=(0.30, 0.30),
                pos=(None, -0.75),
            ),
        ))

        # distractors
        num_distr = np.random.randint(1, 4)
        for i in range(num_distr):
            cfgs.append(dict(
                name=f"distr_counter_{i+1}",
                obj_groups="all",
                placement=dict(
                    fixture=self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(1.0, 0.50),
                    pos=(None, -1.0),
                    offset=(0.0, 0.10),
                ),
            ))

        return cfgs


class CloseDrawer(ManipulateDrawer):
    def __init__(self, *args, **kwargs):        
        super().__init__(behavior="close", *args, **kwargs)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="drawer_obj",
            obj_groups="all",
            graspable=True,
            max_size=(None, None, 0.10),
            placement=dict(
                fixture=self.drawer,
                size=(0.30, 0.30),
                pos=(None, -0.75),
                offset=(0, -self.drawer.size[1] * 0.55)
            ),
        ))

        # distractors
        num_distr = np.random.randint(1, 4)
        for i in range(num_distr):
            cfgs.append(dict(
                name=f"distr_counter_{i+1}",
                obj_groups="all",
                placement=dict(
                    fixture=self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(1.0, 0.50),
                    pos=(None, -1.0),
                    offset=(0.0, 0.10),
                ),
            ))

        return cfgs

