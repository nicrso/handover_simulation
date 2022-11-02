from turtle import position
import pybullet as p 
import pybullet_data
import numpy as np
import time
import os

import camera
from assets import getURDFPath
from control import get_movej_trajectory
from igibson.objects.articulated_object import URDFObject
from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path

class TableEnv:
    def __init__(self, simulator, gui=True):

        #load table
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #/home/nic/miniconda3/envs/sim/lib/python3.7/site-packages/pybullet_data
        # p.loadURDF("table/table.urdf", basePosition=[1.2, 0.0, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071])

        #Load tables
        scene_objects={}

        table_objects_to_load = {
            "table_1": {
                "category": "breakfast_table",
                "model": "5f3f97d6854426cfb41eedea248a6d25",
                "pos": [1.1, 0.0, 0.4],
                "orn": [0, 0, 80],
            },
        }

        for obj in table_objects_to_load.values():
            category = obj["category"]

            if category in scene_objects:
                scene_objects[category] += 1
            else:
                scene_objects[category] = 1

            # Get the path for all models of this category
            category_path = get_ig_category_path(category)

            # If the specific model is given, we use it. If not, we select one randomly
            if "model" in obj:
                model = obj["model"]
            else:
                model = np.random.choice(os.listdir(category_path))

            # Create the full path combining the path for all models and the name of the model
            model_path = get_ig_model_path(category, model)
            filename = os.path.join(model_path, model + ".urdf")

            # Create a unique name for the object instance
            obj_name = "{}_{}".format(category, scene_objects[category])

            # Load the specs of the object categories, e.g., common scaling factor
            avg_category_spec = get_ig_avg_category_specs()

            # Create and import the object
            simulator_obj = URDFObject(
                filename,
                name=obj_name,
                category=category,
                model_path=model_path,
                avg_obj_dims=avg_category_spec.get(category),
                fit_avg_dim_volume=True,
                texture_randomization=False,
                overwrite_inertial=True,
            )
            simulator.import_object(simulator_obj)
            simulator_obj.set_position_orientation(obj["pos"], quat_from_euler(obj["orn"]))

            #set bounds for workplace 
            self.object_ids = list()

            # 3D workspace for table   EDIT 
            self._workspace1_bounds = np.array([
                [1.0, 1.2],  # 3x2 rows: x,y,z cols: min,max
                [-0.1, 0.1],
                [0.5, 1.0]
            ])

        
        # 4 load camera
        self.camera = camera.Camera(
            image_size=(128, 128),
            near=0.01,
            far=10.0,
            fov_w=80
        )
        # camera_target_position = (self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        # camera_target_position[2] = 0.4
        # camera_distance = np.sqrt(((np.array([0.0, 0.0, 1.0]) - camera_target_position)**2).sum()) / 1.7
        # self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=camera_target_position,
        #     distance=camera_distance,
        #     yaw=90,
        #     pitch=-70,
        #     roll=0,
        #     upAxisIndex=2,
        # )

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.8, 0, 0.67],
            cameraTargetPosition=[1.10, 0, 0.5],
            cameraUpVector=[0, 0, 1]
        )
    
    def load_objects(self, name_list, type, seed=None):
        rs = np.random.RandomState(seed=seed)
        for name in name_list:
            urdf_path = getURDFPath(name, type)
            print(urdf_path)
            # position, orientation = self.get_random_pose(rs)
            position, orientation = self.get_fixed_pose()
            obj_id = p.loadURDF(urdf_path, 
                position, p.getQuaternionFromEuler(orientation))
            self.object_ids.append(obj_id)
        self.step_simulation(1e3)

    def get_fixed_pose(self):
        low = self._workspace1_bounds[:,0].copy()
        low[-1] += 0.2
        high = self._workspace1_bounds[:,1].copy()
        high[-1] += 0.2
        position = (high + low) / 2
        orientation = [1, 1, 0]
        return position, orientation
    
    def get_random_pose(self, rs):
        low = self._workspace1_bounds[:,0].copy()
        low[-1] += 0.2
        high = self._workspace1_bounds[:,1].copy()
        high[-1] += 0.2
        position = rs.uniform(low, high, size=3)
        orientation = rs.uniform(-np.pi, np.pi,size=3)
        return position, orientation

    def reset_objects(self, seed=None):
        rs = np.random.RandomState(seed=seed)
        for obj_id in self.object_ids:
            position, orientation = self.get_random_pose(rs)
            p.resetBasePositionAndOrientation(
                obj_id, position, p.getQuaternionFromEuler(orientation))
        self.step_simulation(1e3)
    
    def remove_objects(self):
        for obj_id in self.object_ids:
            p.removeBody(obj_id)
        self.object_ids = list()

    def step_simulation(self, num_steps):
        for i in range(int(num_steps)):
            p.stepSimulation()

    def observe(self):
        rgb_obs, depth_obs, mask_obs = camera.make_obs(self.camera, self.view_matrix)

        return rgb_obs, depth_obs, mask_obs