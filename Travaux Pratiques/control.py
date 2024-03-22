""" A set of robotics control functions """

import random
import numpy as np

from scipy.signal import convolve


KP_SPEED = 0.001
KP_ROTATION = 0.2

FOV = np.deg2rad(240)
KERNEL = np.ones(61)

prev_target = 0.0


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """

    global prev_target

    angles = lidar.get_ray_angles()
    distances = lidar.get_sensor_values()

    in_fov = (angles < 0.5 * FOV) & (-0.5 * FOV < angles)

    index = np.argmax(convolve(distances[in_fov], KERNEL, mode="same"))

    target = angles[in_fov][index]
    dfront = distances[in_fov][index]

    delta = np.abs(target - prev_target)

    if np.abs(target) < np.deg2rad(5.0) and delta < np.deg2rad(5.0):
        speed = np.clip(KP_SPEED * dfront, -1.0, 1.0)
        rotation = 0.0

    else:
        speed = 0.0
        rotation = np.clip(KP_ROTATION * target, -1.0, 1.0)

    prev_target = target

    return {"forward": speed, "rotation": rotation}


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    command = {"forward": 0,
               "rotation": 0}

    return command
