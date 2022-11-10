import argparse
from multiprocessing.process import BaseProcess
import os
from re import S
import time
from unicodedata import category
import robot
from utils import load_from_pkl, degenerate_masks, camera_to_world
from control import accurate_calculate_inverse_kinematics

import numpy as np
import pybullet as p
import pybullet_data
import pickle as pkl

from PIL import Image

import camera
from image import write_rgb, write_depth, write_mask

import igibson

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.articulated_object import URDFObject
from igibson.objects.visual_marker import VisualMarker
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.external.pybullet_tools.utils import (
    set_joint_positions,
)

from common import get_splits
from dataset import VoxelDataset, show_voxel_texture

from table_env import TableEnv
osp = os.path

def main(selection="user", headless=False, short_exec=False):
    """
    Example of usage of inverse kinematics solver
    This is a pybullet functionality but we keep an example because it can be useful and we do not provide a direct
    API from iGibson
    """

    parser = argparse.ArgumentParser(description='Handover Simulation Script')
    parser.add_argument('--seed', type=int, default=10000000,
        help='random seed for empty_bin task')
    parser.add_argument('--type', default='use',
        help='which object type to load: ycb, handoff, or use')
    parser.add_argument("--pose", default="standing",
        help="Sitting or Standing Initial Human Pose")
    args = parser.parse_args()

    #Get dataset for objects to grasp 
    data_dir = osp.join('data', 'voxelized_meshes')
    kwargs = dict(data_dir=data_dir, instruction=args.type, train=True, random_rotation=0, n_ensemble=-1, test_only=False)
    
    dset = VoxelDataset(grid_size=64, **kwargs)
    dset_names = list(dset.filenames.keys())

    # Create simulator, scene and robot (Fetch)
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5, skybox_size=0)

    s = Simulator(mode="gui_interactive", image_height=1024, image_width=1024, rendering_settings=settings, use_pb_gui=True)

    scene = EmptyScene()
    s.import_scene(scene)

    #set up fetch with differentiable drive controller 
    fetch_wrapper = robot.fetch_wrapper(s)
    fetch = fetch_wrapper.fetch

    # Set robot base
    fetch.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
    fetch.reset()
    fetch.tuck()
    fetch.keep_still()

    #load table env 
    env = TableEnv(s)

    #load human model 
    if args.pose=="standing":
        human = p.loadURDF("assets/humans/Darion/urdf/darion_standing.urdf", basePosition=[-2,0,0.07], baseOrientation=p.getQuaternionFromEuler([1.6,0,1.5]), useFixedBase=1)
    elif args.pose=="sitting":
        chair = p.loadURDF("assets/chair/mobility.urdf",basePosition=[-2.0,0,0.41], baseOrientation=p.getQuaternionFromEuler([0,0,3.14]), useFixedBase=1, globalScaling=0.5)
        human = p.loadURDF("assets/humans/Darion/urdf/darion_sitting.urdf", basePosition=[-1.85,0,-0.34], baseOrientation=p.getQuaternionFromEuler([1.6,0,1.5]), useFixedBase=1)

    #load many different objects 
    obj_names = get_splits(args.type)['train']
    obj_names = [i for i in obj_names if i[len(args.type)+1:] in dset_names]
    print(obj_names)
    obj_names = obj_names[:10]

    obj_ids = list()

    ignore_objects = []
    ignore_objects.append("use_door_knob")
    ignore_objects.append("use_flute")


    marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.01)
    s.import_object(marker)

    for name_idx, name in enumerate(obj_names):
        print('Picking: {}'.format(name))
        print(name_idx)

        if name in ignore_objects:
            continue

        #remove objects 
        env.remove_objects()

        env.load_objects([name], args.type)

        #move robot to the table
        action = np.zeros(fetch.action_dim)
        action[0] = 1

        

        #move to position 0.8
        #n_steps'
        des_x = 0.5
        n_steps = 10000
        for i in range(n_steps):
            fetch.set_position_orientation([i/(n_steps/des_x), 0.0, 0.0], [0, 0, 0, 1])

        #get RGB-D data
        rgb_obs, depth_obs, mask_obs = env.observe()

        img_list = env.observe(revolve=True)
        for i,img in enumerate(img_list):
            im = Image.fromarray(img)
            im.save("./data/sim_images/"+name+"_"+str(i)+'.jpg')

        #get human grasp on object
        # pc_filename = "data/predictions/" + name[4:] + "_use_voxnet_diversenet_preds.pkl"

        # pc_filename = osp.expanduser(pc_filename)
        # with open(pc_filename, 'rb') as f:
        #     d = pkl.load(f)  
        # geom, preds = load_from_pkl(d)            
        # geom = geom[0]

        #move robot to table 

        
        #show human grasps 
        #ignore degenerate grasps
        # preds = degenerate_masks(preds)

        #visualize predictions for the objects

        #save scene to numpy array
        # save_dict = {'depth': depth_obs, 'K': env.camera.intrinsic_matrix, 'segmap': mask_obs,'rgb': rgb_obs}
        # np.save('table_'+name+'.npy', save_dict)


        #predict robot grasp 
        #get pick pose from coord, angle, vis_img
        
        robo_grasps_data = np.load("./data/robot_grasp_predictions/predictions_table_" + name + ".npz", allow_pickle=True)
        
        robo_grasps = robo_grasps_data["pred_grasps_cam"].item()[6]
        scores = robo_grasps_data["scores"].item()[6]
        # contact_pts = robo_grasps_data["contact_pts"]

        top_grasp = robo_grasps[np.argmax(scores)]

        top_grasp_coords = camera_to_world(env.camera.view_matrix, top_grasp)[:3, 3]

        print(top_grasp_coords)

        # for _ in range(10000):

        marker.set_position([1.1, 0, 0.5])

        # joint_pos = accurate_calculate_inverse_kinematics(
        #     fetch_wrapper.robot_id, fetch, fetch.eef_links[fetch.default_arm].link_id, [1.1, 0, 0.5], 0.1, 100,
        #     fetch_wrapper.min_limits, fetch_wrapper.max_limits, fetch_wrapper.joint_range, fetch_wrapper.rest_position, fetch_wrapper.joint_damping, fetch_wrapper.arm_joint_ids, 
        #     fetch_wrapper.robot_arm_indices
        # )

        # if joint_pos is not None and len(joint_pos) > 0:
        #     fetch_wrapper.move_joints(s, joint_pos)
            # set_joint_positions(fetch_wrapper.robot_id, fetch_wrapper.arm_joint_ids, joint_pos)

        for _ in range(100):
            s.step()

        #move fetch arm back
        # fetch_wrapper.move_joints_to_tuck(s)
        fetch.tuck()

        # #move to human 
        n_steps = 10000
        for i in range(n_steps):
            fetch.set_position_orientation([0.5-(i/(n_steps/des_x)), 0.0, 0.0], [0, 0, 0, 1])
        fetch.keep_still()

    # visualize human model view
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[-1.3, 0, 1.8],
        cameraTargetPosition=[0.5,0,1.0],
        cameraUpVector=[0, 0, 1]
    )

    human_camera = camera.Camera(
        image_size=(128,128),
        near=0.01,
        far=20.0,
        fov_w=90
    )

    rgb_obs, _, _ = camera.make_obs(human_camera, viewMatrix)  
    write_rgb(rgb_obs, "person_view.jpg")

    fetch_wrapper.teleop(s)

if __name__ == "__main__":
    main()
