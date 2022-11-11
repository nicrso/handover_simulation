import numpy as np
from scipy.interpolate import interp1d
import pybullet as p
import random 
import numpy as np 
import logging
import math 
import argparse

from igibson.external.pybullet_tools.utils import set_joint_positions, get_sample_fn
from igibson.utils.utils import l2_distance

MAX_ITERS = 10000
delta_q = 0.3

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)

def move_robot_base(robot, robot_id, dest=(0,0), n_steps=10000):

    pos, quat = p.getBasePositionAndOrientation(robot_id)
    pos = np.array(pos)
    for i in range(n_steps):
        x = i/(n_steps/dest[0]) if dest[0] != 0 else 0.0
        y = i/(n_steps/dest[1]) if dest[1] != 0 else 0.0
        robot.set_position_orientation([x, y, 0.0]+pos, quat)

def rotate_robot_base(robot, robot_id, rot=3.14, n_steps=10000):

    pos, quat_prev = p.getBasePositionAndOrientation(robot_id)
    quat_prev = np.array(p.getEulerFromQuaternion(quat_prev))
    for i in range(n_steps):
        quat = np.array(p.getQuaternionFromEuler([0, 0, i/(n_steps/rot)+quat_prev[2]]))
        robot.set_position_orientation(pos, quat)

    return 

def accurate_calculate_inverse_kinematics(robot_id, fetch, eef_link_id, curr_pos, target_pos, threshold, max_iter, min_limits, max_limits, joint_range, rest_position, joint_damping, arm_joint_ids, robot_arm_indices):
    print("IK solution to end effector position {}".format(target_pos))
    # Save initial robot pose
    state_id = p.saveState()

    max_attempts = 100
    solution_found = False
    joint_poses_list = []
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
                joint_poses_list.append(joint_poses)
                break
            logging.debug("Dist: " + str(dist))
            it += 1

        if solution_found:
            print("Solution found at iter: " + str(it) + ", residual: " + str(dist))
            continue
        else:
            print("Attempt failed. Retry")
            joint_poses = None

    #manually rank the the samples based on the L2 distance that is closest to the rest pose.

    p.restoreState(state_id)

    #return closest pose based on l2 distance 
    joint_poses_list = sorted(joint_poses_list, key=lambda x: l2_distance(curr_pos, x))

    print(len(joint_poses_list))
    return joint_poses_list[0]

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    Vertex = [q_init]
    Edges = {}    
    
    for i in range(1, MAX_ITERS):
        q_rand = semiRandomSample(steer_goal_p, q_goal)
        q_nearest = nearest(Vertex, q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)        
        
        if obstacleFree(q_new, env):
    #       V = Union(V, {q_new})
            Vertex.append(q_new)
    #       E = Union(E, {(q_nearest, q_new)})
            Edges[tuple(q_new)] = tuple(q_nearest)            #visualize: create a line between q_new and q_nearest
            visualize_path(q_new, q_nearest, env)            
            
            if dist(q_goal, q_new) < delta_q:                
                
                visualize_path(q_new, q_goal, env)
    #           V = Union(V, {q_goal})
                Vertex.append(q_goal)
    #           E = Union(E, {(q_new, q_goal)})
                Edges[tuple(q_goal)] = tuple(q_new)                
                
                path = []                
                path.append(np.asarray(q_goal))                
                curr_parent = q_goal                
                
                #trace back to q_init from q_goal
                while not np.array_equal(curr_parent, q_init):                    
                    curr_parent = np.asarray(Edges[tuple(curr_parent)])
                    path.append(curr_parent)                
                    
                path.reverse()                
                
                return path    
    # ==================================
    return None

def nearest(Vertex, q_rand):
    #given a point in CSpace, find vertex on tree that is closest to that point    
    min = 10000000
    argMin = 0    
    
    for node in Vertex:
        d = dist(node, q_rand)
        if d < min:
            min = d
            argMin = node    
            
    return argMin

def semiRandomSample(steer_goal_p, q_goal):
    #with probability steer_goal_p return goal config, else return uniform sample    
    
    rand = np.random.uniform(-2*np.pi, 2*np.pi, 6)
    choices = [rand, q_goal]
    choice = np.random.choice([0, 1], 1, p=[1-steer_goal_p, steer_goal_p])
    return choices[choice[0]]
    
def steer(q_nearest, q_rand, delta_q):    
    
    distance = dist(q_nearest, q_rand)    
    
    if distance <= delta_q:
        q_new = q_nearest
    else:
        #return vector along q_rand-q_nearest that is delta_q away from q_nearest
        q_new = q_nearest + (q_rand - q_nearest)*(delta_q/distance)    
        
    return q_new

def obstacleFree(q, env):
    return not env.check_collision(q)    #we assume delta+q is small enough that we only need to check obstableFree on x_new
    #check if the path from x to y is collision free

def dist(q1, q2):
    #return euclidean distance
    return np.linalg.norm(q1-q2)

def execute_path(path_conf, env):
    """
    :param path_conf: list of configurations (joint angles)
    """
     
    pos5 = p.getLinkState(env.robot_body_id, 9)[0]  
    

    for position in path_conf:
        #move arm to new position
        env.move_joints(position)
        state = p.getLinkState(env.robot_body_id, 9)[0]
        
    env.open_gripper()
    env.close_gripper()    
    
    for i in range(len(path_conf)-1, -1, -1):
        env.move_joints(path_conf[i])    
    
    # ==================================


def get_grasp_position_angle(object_id):
    """
    Get position and orientation (yaw in radians) of the object
    :param object_id: object id
    """
    position, grasp_angle = np.zeros(3), 0
     
    position, quat = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(quat)[2]   
    return position, grasp_angle


def get_trapezoid_phase_profile(
        dt=0.01,
        start_phase=0.0,
        end_phase=1.0,
        speed=1.0, 
        acceleration=1.0, 
        start_padding=0.0, 
        end_padding=0.0,
        dtype=np.float64):
    
    # calculate duration
    assert(end_phase > start_phase)
    total_travel = end_phase - start_phase

    t_cruise = None
    t_acc = None
    max_speed = None
    tri_max_speed = np.sqrt(acceleration * total_travel)
    if tri_max_speed <= speed:
        # triangle
        t_acc = total_travel / tri_max_speed
        t_cruise = 0
        max_speed = tri_max_speed
    else:
        # trapozoid
        t_acc = speed / acceleration
        tri_travel = t_acc * speed
        t_cruise = (total_travel - tri_travel) / speed
        max_speed = speed

    duration = start_padding + end_padding + t_acc * 2 + t_cruise
    key_point_diff_arr = np.array([
        start_padding, t_acc, t_cruise, t_acc, end_padding], dtype=dtype)
    key_point_time_arr = np.cumsum(key_point_diff_arr)

    all_time_steps = np.linspace(0, duration, int(np.ceil(duration / dt)), dtype=dtype)
    phase_steps = np.zeros_like(all_time_steps)

    # start_padding
    mask_idxs = np.flatnonzero(all_time_steps < key_point_time_arr[0])
    if len(mask_idxs) > 0:
        phase_steps[mask_idxs] = start_phase
    
    # acceleartion
    mask_idxs = np.flatnonzero(
        (key_point_time_arr[0] <= all_time_steps) 
        & (all_time_steps <= key_point_time_arr[1]))
    if len(mask_idxs) > 0:
        phase_steps[mask_idxs] = acceleration / 2 * np.square(
            all_time_steps[mask_idxs] - key_point_time_arr[0])
    acc_dist = acceleration / 2 * (t_acc ** 2)
    
    # cruise
    mask_idxs = np.flatnonzero(
        (key_point_time_arr[1] < all_time_steps) 
        & (all_time_steps < key_point_time_arr[2]))
    if len(mask_idxs) > 0:
        phase_steps[mask_idxs] = max_speed * (
            all_time_steps[mask_idxs] - key_point_time_arr[1]) + acc_dist
    cruise_dist = t_cruise * max_speed

    # deceleration
    mask_idxs = np.flatnonzero(
        (key_point_time_arr[2] <= all_time_steps) 
        & (all_time_steps <= key_point_time_arr[3]))
    if len(mask_idxs) > 0:
        curr_time_steps = all_time_steps[mask_idxs] - key_point_time_arr[2]
        phase_steps[mask_idxs] = max_speed * curr_time_steps \
            - acceleration / 2 * np.square(curr_time_steps) \
            + acc_dist + cruise_dist
    
    # end_padding
    int_end_phase = acc_dist * 2 + cruise_dist
    mask_idxs = np.flatnonzero(key_point_time_arr[3] <= all_time_steps)
    if len(mask_idxs) > 0:
        phase_steps[mask_idxs] = int_end_phase
        
    return phase_steps


def get_movej_trajectory(j_start, j_end, acceleration, speed, dt=0.001):
    assert(acceleration > 0)
    assert(speed > 0)

    j_delta = j_end - j_start
    j_delta_abs = np.abs(j_delta)
    j_delta_max = np.max(j_delta_abs)

    if j_delta_max != 0:
        # compute phase parameters
        phase_vel = speed / j_delta_max
        phase_acc = acceleration * (phase_vel)

        phase = get_trapezoid_phase_profile(dt=dt,
            speed=phase_vel, acceleration=phase_acc)
        
        interp = interp1d([0,1], [j_start, j_end], 
            axis=0, fill_value='extrapolate')
        j_traj = interp(phase)
    else:
        j_traj = np.array([j_start, j_end])

    assert(np.allclose(j_traj[0], j_start))
    assert(np.allclose(j_traj[-1], j_end))
    return j_traj



