import os 
import time
import numpy as np
import pybullet as p
import igibson
from igibson.robots.fetch import Fetch
from igibson.utils.utils import parse_config
from igibson.objects.visual_marker import VisualMarker
from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
)
from control import accurate_calculate_inverse_kinematics

class fetch_wrapper():
    def __init__(self, s):

        #Load config 
        config = parse_config(os.path.join(igibson.configs_path, "fetch_reaching.yaml"))

        #Load fetch 
        robot_config = config["robot"]
        robot_config.pop("name")

        self.fetch = Fetch(**robot_config)

        s.import_object(self.fetch)

        self.arm_default_joint_positions = (
            0.10322468280792236,
            -11.814019864768982,
            11.5178184935241699,
            10.8189625336474915,
            15.200358942909668,
            10.9631312579803466,
            -10.2862852996643066,
            10.0008453550418615341,
        )

        self.robot_default_joint_positions = (
            [0.0, 0.0]
            + [self.arm_default_joint_positions[0]]
            + [0.0, 0.0]
            + list(self.arm_default_joint_positions[1:])
            + [1.01, 1.01]
        )

        self.robot_joint_names = [
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
        self.arm_joints_names = [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        body_ids = self.fetch.get_body_ids()
        assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
        self.robot_id = body_ids[0]

        # Indices of the joints of the arm in the vectors returned by IK and motion planning (excluding wheels, head, fingers)
        self.robot_arm_indices = [self.robot_joint_names.index(arm_joint_name) for arm_joint_name in self.arm_joints_names]

        # PyBullet ids of the joints corresponding to the joints of the arm
        self.arm_joint_ids = joints_from_names(self.robot_id, self.arm_joints_names)
        self.all_joint_ids = joints_from_names(self.robot_id, self.robot_joint_names)

        set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)

        # Define the limits (max and min, range), and resting position for the joints, including the two joints of the
        # wheels of the base (indices 0 and 1), the two joints for the head (indices 3 and 4) and the two joints of the
        # fingers (indices 12 and 13)
        self.max_limits = get_max_limits(self.robot_id, self.all_joint_ids)
        self.min_limits = get_min_limits(self.robot_id, self.all_joint_ids)
        self.rest_position = self.robot_default_joint_positions
        self.joint_range = list(np.array(self.max_limits) - np.array(self.min_limits))
        self.joint_range = [item + 1 for item in self.joint_range]
        self.joint_damping = [0.1 for _ in self.joint_range]

    # def move_above_table(s):


    def move_joints(self, s, target_joint_state, speed=0.02):
        """
            Move the robot arm to the specified joint state
        """

        assert(len(target_joint_state)>0)

        p.setJointMotorControlArray(
            self.robot_id, 
            self.arm_joint_ids,
            p.POSITION_CONTROL,
            target_joint_state,
            positionGains=speed*np.ones(len(self.arm_joint_ids))
        )

        timeout_t0 = time.time()

        while True:
            current_joint_state = [
                p.getJointState(self.robot_id,i)[0] for i in self.arm_joint_ids
            ]
            if all([
                np.abs(current_joint_state[i]-target_joint_state[i]) > 1e-3
                for i in range(len(self.arm_joint_ids))
            ]):
                break
            if time.time()-timeout_t0>5:
                print("Timeout: robot motion taking too long.")
                self.fetch.tuck()
                break
            s.step()
    
    def move_joints_to_tuck(self, s, speed=0.02):
        """
            Move the robot arm to the specified joint state
        """

        p.setJointMotorControlArray(
            self.robot_id, 
            self.all_joint_ids,
            p.POSITION_CONTROL,
            self.fetch.tucked_default_joint_pos,
            positionGains=speed*np.ones(len(self.all_joint_ids))
        )

        timeout_t0 = time.time()

        while True:
            current_joint_state = [
                p.getJointState(self.robot_id,i)[0] for i in self.all_joint_ids
            ]
            if all([
                np.abs(current_joint_state[i]-self.fetch.tucked_default_joint_pos[i]) > 1e-3
                for i in range(len(self.all_joint_ids))
            ]):
                break
            if time.time()-timeout_t0>5:
                print("Timeout: robot motion taking too long.")
                self.fetch.tuck()
                break
            s.step()

    def teleop(self, s):

        threshold = 0.1
        max_iter = 100

        self.fetch.tuck()
        marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
        s.import_object(marker)

        x, y, z = self.fetch.get_eef_position()

        marker.set_position([x, y, z])

        self.print_message()
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
                        self.robot_id, self.fetch, self.fetch.eef_links[self.fetch.default_arm].link_id, [x, y, z], threshold, max_iter,
                        self.min_limits, self.max_limits, self.joint_range, self.rest_position, self.joint_damping, self.arm_joint_ids, 
                        self.robot_arm_indices
                    )
                    if joint_pos is not None and len(joint_pos) > 0:
                        print("Solution found. Setting new arm configuration.")
                        set_joint_positions(self.robot_id, self.arm_joint_ids, joint_pos)
                        self.print_message()
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
            self.fetch.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
            self.fetch.keep_still()
            s.step()

        s.disconnect()


    def print_message(self):
        print("*" * 80)
        print("Move the marker to a desired position to query IK and press SPACE")
        print("Up/Down arrows: move marker further away or closer to the robot")
        print("Left/Right arrows: move marker to the left or the right of the robot")
        print("z/x: move marker up and down")
        print("q: quit")
