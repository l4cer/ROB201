""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np

from utils import OccupancyGrid

from constants import *


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """

        angles = lidar.get_ray_angles()
        distances = lidar.get_sensor_values()

        x_world = pose[0] + distances * np.cos(pose[2] + angles)
        y_world = pose[1] + distances * np.sin(pose[2] + angles)

        x_map, y_map = self.grid.world2grid(x_world, y_world)

        x = np.floor(x_map).astype(int)
        y = np.floor(y_map).astype(int)

        mask = (0 <= x) & (x < self.grid.size_x) & (0 <= y) & (y < self.grid.size_y)

        score = np.sum(self.grid.occupancy[x[mask], y[mask]])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        angle = np.arctan2(odom_pose[1], odom_pose[0])
        distance = np.linalg.norm(odom_pose[:2])

        corrected_pose = [
            odom_pose_ref[0] + distance * np.cos(odom_pose_ref[2] + angle),
            odom_pose_ref[1] + distance * np.sin(odom_pose_ref[2] + angle),
            odom_pose_ref[2] + odom_pose[2]
        ]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose, N = 125):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """

        best_pose_ref = self.odom_pose_ref
        best_score = self._score(lidar, self.get_corrected_pose(raw_odom_pose))

        counter = 0

        while counter < N:
            delta = np.random.normal([0.0, 0.0, 0.0], [5.0, 5.0, 0.15])
            new_ref = best_pose_ref + delta

            corrected = self.get_corrected_pose(
                raw_odom_pose, odom_pose_ref=new_ref)

            score = self._score(lidar, corrected)

            if best_score is None or score > best_score:
                best_score = score
                best_pose_ref = new_ref

                counter = 0

            else:
                counter += 1

        self.odom_pose_ref = best_pose_ref

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """

        angles = lidar.get_ray_angles()
        distances = lidar.get_sensor_values()

        cos = np.cos(pose[2] + angles)
        sin = np.sin(pose[2] + angles)

        x_free_space = pose[0] + (distances - WALL_SPACING) * cos
        y_free_space = pose[1] + (distances - WALL_SPACING) * sin

        x_wall_start = pose[0] + (distances - WALL_WIDTH / 2) * cos
        y_wall_start = pose[1] + (distances - WALL_WIDTH / 2) * sin

        x_wall_end = pose[0] + (distances + WALL_WIDTH / 2) * cos
        y_wall_end = pose[1] + (distances + WALL_WIDTH / 2) * sin

        for index in range(len(angles)):
            self.grid.increment_line(pose[0],
                                     pose[1],
                                     x_free_space[index],
                                     y_free_space[index],
                                     LOG_PROB_FREE_SPACE)

            self.grid.increment_line(x_wall_start[index],
                                     y_wall_start[index],
                                     x_wall_end[index],
                                     y_wall_end[index],
                                     LOG_PROB_OCCUPIED)

        self.grid.occupancy = np.clip(
            self.grid.occupancy, MIN_LOG_PROB, MAX_LOG_PROB)

        #self._score(lidar, pose)
        self.grid.display(pose)
