from robosuite.utils.mjcf_utils import array_to_string as a2s, string_to_array as s2a, CustomMaterial, xml_path_completion
from robosuite.models.objects import CompositeBodyObject, BoxObject
import numpy as np

import robocasa


#creates num_windows side-by side where each subsequent is created in increasing X-direction
#size represents the size the group


class Window(CompositeBodyObject):
    def __init__(self, name, size, ofs = None, pos=None, quat=None, window_bak="textures/others/bk7.png",  texture="textures/flat/white.png", trim_th=0.02, trim_size=0.015, num_windows=1):
        self.size = size
        self.origin_offset = [0, 0, 0]
        self.window_size = [size[0]/num_windows, size[1], size[2]]
        self.texture = xml_path_completion(texture, robocasa.models.assets_root)
        self.window_bak = xml_path_completion(window_bak, robocasa.models.assets_root)
        self.num_windows = num_windows
        self.pos = [0,0,0] if pos is None else pos
        tex_attrib = {
            "type": "2d"
        }
        #now do not consider passed in quat bc havent combined 
        #wanted quat with the rotation applied to the window
        # self.quats = quat
        self.quats = []
        self.trim_size = trim_size
        self.trim_th = trim_th
        trim_mat_attrib = {
            "texrepeat": "4 4",
            "specular": "0.1",
            "shininess": "0.1",
            "texuniform": "true"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.1",
            "shininess": "0.1",
            "texuniform": "true"
        }
        self.trim_mat = CustomMaterial(
            texture=self.texture,
            tex_name="panel_tex",
            mat_name="panel_mat",
            tex_attrib=tex_attrib,
            mat_attrib=trim_mat_attrib,
            shared=True,
        )
        self.window_mat = CustomMaterial(
            texture=self.window_bak,
            tex_name="blurred_bak",
            mat_name="window_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=True,
        )
        
        self.center = np.array([0, 0, 0])
        self.scale = 1.0
        self.num_windows = num_windows
        self.ofs = ofs if ofs is not None else [0.0, 0.0, 0.0]
        self.ofs = np.array(self.ofs)

        

        self.create_window()
        
        super().__init__(
            name=name,
            objects= self.objects,
            object_locations=self.positions,
            object_quats=self.quats,
            joints=None
        )

#change to create objects then create positions


    def create_window(self):
        x, y, z = self.window_size
        x, y, z = x / 2, y/2, z / 2
        door_th = y - self.trim_th
        sizes = [
            [self.trim_size, self.trim_th, z],
            [self.trim_size, self.trim_th, z],
            [x - 2 * self.trim_size, self.trim_th, self.trim_size],
            [x - 2 * self.trim_size, self.trim_th, self.trim_size],
            [self.trim_size/2, self.trim_th, z],
            [x - 2 * self.trim_size, self.trim_th, self.trim_size/2]


        ] * self.num_windows

        #switch y and z sizes because we will rotate
        #we are doing this because textures are only displayed on the z-parallel part of the box
        sizes.append([x*self.num_windows, z, door_th])


        base_names = ["trim_left", "trim_right", "trim_top", "trim_bottom", "vert_trim", "horiz_trim"]
        names = [f"{name}_{i}" for i in range(self.num_windows) for name in base_names]
        names.append("door")


        offsets = self._get_window_offsets()
        positions = []


        for offset in offsets:
            positions.extend(
                    [
                    np.array([-x + self.trim_size + offset, -0.0045, 0]),
                    np.array([x - self.trim_size + offset, -0.0045, 0]),
                    np.array([offset, -0.0045, z - self.trim_size]),
                    np.array([offset, -0.0045, -z + self.trim_size]),
                    np.array([offset, -0.0045, 0]),
                    np.array([offset, -0.0045, 0])
                ]
            )
        positions.append(np.array([0, 0, 0]))
        
        objects = []


        for obj_name, size in zip(names, sizes):
            if "door" in obj_name:
                new_obj = BoxObject(name=obj_name, size=np.array(size), material=self.window_mat)
            else:
                new_obj = BoxObject(name=obj_name, size=np.array(size), material=self.trim_mat)
            objects.append(new_obj)
       
        self.objects = objects
        self.positions = [position + self.ofs for position in positions]
        self.quats = [None] * (len(objects)-1)
        self.quats.append([ 0, 0, 0.7071081, 0.7071055 ]) 


        

    def _get_window_offsets(self):
        x = self.window_size[0]/2
        start = (-self.size[0]/2) + x
        end = self.size[0]/2 - x
        offsets = np.linspace(start, end, self.num_windows)
        return offsets


    def set_pos(self, pos):
        self.pos = pos
        self._obj.set("pos", a2s(pos))
    
    def update_state(self, env):
        return
    
    @property
    def nat_lang(self):
        return "windows"
    
    @property
    def rot(self):
        rot = s2a(self._obj.get("euler", "0.0 0.0 0.0"))
        return rot[2]
  

  #sample the start and end offsets


class FramedWindow(Window):
    def __init__(self, name, size, ofs = None, pos=None, quat=None, window_bak="textures/others/bk7.png", texture="textures/flat/white.png", trim_th=0.02, trim_size=0.015, num_windows=1, frame_width=0.05):
        self.frame_width = frame_width
        super().__init__(name=name, size=size,ofs=ofs, pos=pos, quat=quat,window_bak=window_bak, texture=texture, trim_th=trim_th, trim_size=trim_size, num_windows=num_windows)


        
    def create_window(self):
        self.window_size = [(self.size[0] - self.frame_width)/self.num_windows, self.size[1], self.size[2]-self.frame_width]
        super().create_window()


        #created the window now add the frame!
        x, y, z = self.window_size
        x, y, z = x / 2, y/2, z / 2
        sizes = [
                [self.size[0]/2,y, self.frame_width/4],
                [self.size[0]/2,y, self.frame_width/4],
                [self.frame_width/4, y, z],
                [self.frame_width/4, y, z]
            ]
        
        names = ["frame_top", "frame_bot", "frame_right", "frame_left"]
        val = (self.size[0] - self.frame_width)/2
        frame_positions = [
                np.array([0, 0.00145,  z+self.frame_width/4]),
                np.array([0, 0.00145 , -z-self.frame_width/4]),
                np.array([val+self.frame_width/4, 0.00145 , 0]),
                np.array([-val - self.frame_width/4, 0.00145 , 0]),
        ]
        new_positions = [frame_pos + self.ofs for frame_pos in frame_positions]
        self.positions.extend(new_positions)
        self.quats.extend([None] * len(new_positions))
    


        for obj_name, size in zip(names, sizes):
            self.objects.append(BoxObject(name=obj_name, size=np.array(size),material=self.trim_mat))
    

    def _get_window_offsets(self):
        x = self.window_size[0]/2
        start = ((self.frame_width-self.size[0])/2) + x
        end = ((self.size[0]-self.frame_width)/2) - x
        offsets = np.linspace(start, end, self.num_windows)
        return offsets


    def set_pos(self, pos):
        self.pos = pos
        self._obj.set("pos", a2s(pos))

