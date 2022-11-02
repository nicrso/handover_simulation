import argparse
from email.mime import image
import logging
from multiprocessing.process import BaseProcess
import os
from re import S
import time
from unicodedata import category
from utils import load_from_pkl, degenerate_masks, camera_to_world

import numpy as np
import pybullet as p
import pybullet_data
import pickle as pkl

from PIL import Image

import camera
from image import write_rgb, write_depth, write_mask

import igibson
from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
)

from igibson.robots.fetch import Fetch
from igibson.objects.visual_marker import VisualMarker
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.articulated_object import URDFObject
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.utils import l2_distance, parse_config

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


    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Create simulator, scene and robot (Fetch)
    config = parse_config(os.path.join(igibson.configs_path, "fetch_reaching.yaml"))

    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5, skybox_size=0)

    s = Simulator(
        mode="gui_interactive", 
        image_width=512, 
        image_height=512,
        rendering_settings=settings, 
        use_pb_gui=True
    )

    scene = EmptyScene()
    s.import_scene(scene)

    #Load fetch 
    robot_config = config["robot"]
    robot_config.pop("name")

    #set up fetch with differentiable drive controller 

    fetch = Fetch(**robot_config)
    s.import_object(fetch)

    body_ids = fetch.get_body_ids()
    assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
    robot_id = body_ids[0]

    arm_default_joint_positions = (
        10.10322468280792236,
        -11.814019864768982,
        11.5178184935241699,
        10.8189625336474915,
        15.200358942909668,
        10.9631312579803466,
        -10.2862852996643066,
        10.0008453550418615341,
    )

    robot_default_joint_positions = (
        [0.0, 0.0]
        + [arm_default_joint_positions[0]]
        + [0.0, 0.0]
        + list(arm_default_joint_positions[1:])
        + [1.01, 1.01]
    )

    robot_joint_names = [
        "r_wheel_joint",
        "l_wheel_joint",
        "torso_lift_joint",
        "head_pan_joint",
        "head_tilt_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        "r_gripper_finger_joint",
        "l_gripper_finger_joint",
    ]
    arm_joints_names = [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]

    # Indices of the joints of the arm in the vectors returned by IK and motion planning (excluding wheels, head, fingers)
    robot_arm_indices = [robot_joint_names.index(arm_joint_name) for arm_joint_name in arm_joints_names]

    # PyBullet ids of the joints corresponding to the joints of the arm
    arm_joint_ids = joints_from_names(robot_id, arm_joints_names)
    all_joint_ids = joints_from_names(robot_id, robot_joint_names)

    set_joint_positions(robot_id, arm_joint_ids, arm_default_joint_positions)

    # Set robot base
    fetch.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
    fetch.reset()
    fetch.tuck()
    fetch.keep_still()

    # Get initial EE position
    x, y, z = fetch.get_eef_position()

    # Define the limits (max and min, range), and resting position for the joints, including the two joints of the
    # wheels of the base (indices 0 and 1), the two joints for the head (indices 3 and 4) and the two joints of the
    # fingers (indices 12 and 13)
    max_limits = get_max_limits(robot_id, all_joint_ids)
    min_limits = get_min_limits(robot_id, all_joint_ids)
    rest_position = robot_default_joint_positions
    joint_range = list(np.array(max_limits) - np.array(min_limits))
    joint_range = [item + 1 for item in joint_range]
    joint_damping = [0.1 for _ in joint_range]

    #load table env 
    env = TableEnv(s)

    #load human model 
    if args.pose=="standing":
        human = p.loadURDF("assets/humans/Darion/urdf/darion_standing.urdf", basePosition=[-2,0,0.07], baseOrientation=p.getQuaternionFromEuler([1.6,0,1.5]), useFixedBase=1)
    elif args.pose=="sitting":
        chair = p.loadURDF("assets/chair/mobility.urdf",basePosition=[-2.0,0,0.41], baseOrientation=p.getQuaternionFromEuler([0,0,3.14]), useFixedBase=1, globalScaling=0.5)
        human = p.loadURDF("assets/humans/Darion/urdf/darion_sitting.urdf", basePosition=[-1.85,0,-0.34], baseOrientation=p.getQuaternionFromEuler([1.6,0,1.5]), useFixedBase=1)

    def accurate_calculate_inverse_kinematics(robot_id, eef_link_id, target_pos, threshold, max_iter):
        print("IK solution to end effector position {}".format(target_pos))
        # Save initial robot pose
        state_id = p.saveState()

        max_attempts = 5
        solution_found = False
        joint_poses = None
        for attempt in range(1, max_attempts + 1):
            print("Attempt {} of {}".format(attempt, max_attempts))
            # Get a random robot pose to start the IK solver iterative process
            # We attempt from max_attempt different initial random poses
            sample_fn = get_sample_fn(robot_id, arm_joint_ids)
            sample = np.array(sample_fn())
            # Set the pose of the robot there
            set_joint_positions(robot_id, arm_joint_ids, sample)

            it = 0
            # Query IK, set the pose to the solution, check if it is good enough repeat if not
            while it < max_iter:

                joint_poses = p.calculateInverseKinematics(
                    robot_id,
                    eef_link_id,
                    target_pos,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                )
                joint_poses = np.array(joint_poses)[robot_arm_indices]

                set_joint_positions(robot_id, arm_joint_ids, joint_poses)

                dist = l2_distance(fetch.get_eef_position(), target_pos)
                if dist < threshold:
                    solution_found = True
                    break
                logging.debug("Dist: " + str(dist))
                it += 1

            if solution_found:
                print("Solution found at iter: " + str(it) + ", residual: " + str(dist))
                break
            else:
                print("Attempt failed. Retry")
                joint_poses = None

        # restoreState(state_id)
        p.removeState(state_id)
        return joint_poses

    threshold = 0.1
    max_iter = 100

    #load many different objects 
    obj_names = get_splits(args.type)['train']
    obj_names = [i for i in obj_names if i[len(args.type)+1:] in dset_names]
    print(obj_names)
    obj_names = obj_names[:10]

    obj_ids = list()

    ignore_objects = []
    ignore_objects.append("use_door_knob")
    ignore_objects.append("use_flute")

    n_attempts = 1

    marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
    s.import_object(marker)

    for name_idx, name in enumerate(obj_names):
        print('Picking: {}'.format(name))
        print(name_idx)

        if name in ignore_objects:
            continue

        #remove objects 
        env.remove_objects()

        for i in range(n_attempts):
            seed = name_idx * 100 + i + 10000

            if i == 0:
                #load objects
                env.load_objects([name], args.type, seed=seed)
            else: 
                #reset objects
                env.reset_objects(seed)

            #get RGB-D data
            rgb_obs, depth_obs, mask_obs = env.observe()

            for _ in range(30):
                s.step()
            
            robo_grasps_data = np.load("./data/robot_grasp_predictions/predictions_table_" + name + ".npz", allow_pickle=True)
            
            robo_grasps = robo_grasps_data["pred_grasps_cam"].item()[6]
            scores = robo_grasps_data["scores"].item()[6]
            # contact_pts = robo_grasps_data["contact_pts"]

            top_grasp = robo_grasps[np.argmax(scores)]

            top_grasp_coords = camera_to_world(env.view_matrix, top_grasp)[:3, 3]

            print(top_grasp_coords)

            for _ in range(10000):

                marker.set_position([top_grasp_coords[0], top_grasp_coords[1], top_grasp_coords[2]])

                joint_pos = accurate_calculate_inverse_kinematics(robot_id, fetch.eef_links[fetch.default_arm].link_id, [top_grasp_coords[0], top_grasp_coords[1], top_grasp_coords[2]], threshold, max_iter=100)

                print(joint_pos.shape)
                if joint_pos is not None and len(joint_pos) > 0:
                    print("Solution found. Setting new arm configuration.")
                    set_joint_positions(robot_id, arm_joint_ids, joint_pos)
                    print_message()

                fetch.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
                fetch.keep_still()
                s.step()


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


    marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
    s.import_object(marker)
    marker.set_position([x, y, z])

    print_message()
    quit_now = False

    while True:
        keys = p.getKeyboardEvents()
        for k, v in keys.items():
            if k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN):
                y -= 0.01
            if k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN):
                y += 0.01
            if k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN):
                x += 0.01
            if k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN):
                x -= 0.01
            if k == ord("z") and (v & p.KEY_IS_DOWN):
                z += 0.01
            if k == ord("x") and (v & p.KEY_IS_DOWN):
                z -= 0.01
            if k == ord(" "):
                print("Querying joint configuration to current marker position")
                joint_pos = accurate_calculate_inverse_kinematics(
                    robot_id, fetch.eef_links[fetch.default_arm].link_id, [x, y, z], threshold, max_iter
                )
                if joint_pos is not None and len(joint_pos) > 0:
                    print("Solution found. Setting new arm configuration.")
                    set_joint_positions(robot_id, arm_joint_ids, joint_pos)
                    print_message()
                else:
                    print(
                        "No configuration to reach that point. Move the marker to a different configuration and try again."
                    )
            if k == ord("q"):
                print("Quit.")
                quit_now = True
                break

        if quit_now:
            break

        marker.set_position([x, y, z])
        fetch.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
        fetch.keep_still()
        s.step()

    s.disconnect()


def print_message():
    print("*" * 80)
    print("Move the marker to a desired position to query IK and press SPACE")
    print("Up/Down arrows: move marker further away or closer to the robot")
    print("Left/Right arrows: move marker to the left or the right of the robot")
    print("z/x: move marker up and down")
    print("q: quit")


if __name__ == "__main__":
    main()
