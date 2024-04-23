""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np

from occupancy_grid import OccupancyGrid


WALL_WIDTH = 2.0
WALL_SPACING = 10.0

MIN_LOG_PROB = -8.0
MAX_LOG_PROB =  8.0


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

        angles = lidar.get_ray_angles() + pose[2]
        distances = lidar.get_sensor_values()

        mask = distances < 300.0

        cos, sin = np.cos(angles), np.sin(angles)

        x = (distances * cos + pose[0])[mask]
        y = (distances * sin + pose[1])[mask]

        x = (x - self.grid.x_min_world) / self.grid.resolution
        y = (y - self.grid.y_min_world) / self.grid.resolution

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)

        x1 = np.ceil(x).astype(int)
        y1 = np.ceil(y).astype(int)

        mp00 = self.grid.occupancy_map[x0, y0]
        mp01 = self.grid.occupancy_map[x0, y1]
        mp10 = self.grid.occupancy_map[x1, y0]
        mp11 = self.grid.occupancy_map[x1, y1]

        tmp = np.power(10.0,
            (y - y0) * ((x - x0) * mp11 + (x1 - x) * mp01) +
            (y1 - y) * ((x - x0) * mp10 + (x1 - x) * mp00)
        )

        score = np.sum(tmp / (tmp + 1.0))

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

        dist = np.linalg.norm(odom_pose[:2])
        angle = np.arctan2(odom_pose[1], odom_pose[0])

        corrected_pose = [
            odom_pose_ref[0] + dist * np.cos(angle + odom_pose_ref[2]),
            odom_pose_ref[1] + dist * np.sin(angle + odom_pose_ref[2]),
            odom_pose_ref[2] + odom_pose[2]
        ]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose, N = 50):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """

        best_score = None
        best_offset = None

        counter = 0

        while counter < N:
            offset = np.random.normal([0.0, 0.0, 0.0], [3.0, 3.0, 0.01])

            corrected = self.get_corrected_pose(
                raw_odom_pose, odom_pose_ref=offset)

            score = self._score(lidar, corrected)

            if best_score is None or score > best_score:
                best_score = score
                best_offset = offset

                counter = 0

            else:
                counter += 1

        self.odom_pose_ref = best_offset

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """

        angles = lidar.get_ray_angles() + pose[2]
        distances = lidar.get_sensor_values()

        cos, sin = np.cos(angles), np.sin(angles)

        x_wall = (distances - WALL_SPACING) * cos + pose[0]
        y_wall = (distances - WALL_SPACING) * sin + pose[1]

        x_bef = (distances - WALL_WIDTH / 2) * cos + pose[0]
        y_bef = (distances - WALL_WIDTH / 2) * sin + pose[1]

        x_aft = (distances + WALL_WIDTH / 2) * cos + pose[0]
        y_aft = (distances + WALL_WIDTH / 2) * sin + pose[1]

        prob = np.log10(0.95 / (1.0 - 0.95))

        for i in range(len(angles)):
            self.grid.add_map_line(
                pose[0], pose[1], x_wall[i], y_wall[i], -3.0)

            self.grid.add_map_line(
                x_bef[i], y_bef[i], x_aft[i], y_aft[i], prob)

        self.grid.occupancy_map = np.clip(
            self.grid.occupancy_map, MIN_LOG_PROB, MAX_LOG_PROB)

        #self._score(lidar, pose)
        self.grid.display_cv(pose)
