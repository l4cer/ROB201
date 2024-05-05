import numpy as np

from place_bot.entities.lidar import Lidar

from utils import Grid

from constants import *


class TinySlam:
    """Simple tiny SLAM object"""

    def __init__(self, grid: Grid, lidar: Lidar) -> None:
        """
        Constructor of the TinySlam class.

        Args:
            grid (Grid): single instance of the Grid class.
            lidar (Lidar): single instance of the Lidar class.
        """

        self.grid: Grid = grid

        self.lidar: Lidar = lidar

        # Origin of the odometry frame in the map frame
        self.odom_origin: np.ndarray = np.array([0.0, 0.0, 0.0])

    def score(self, pose: np.ndarray) -> float:
        """
        Computes the sum of log probabilities of laser end points in the map.

        Args:
            pose (np.ndarray): robot pose in format [x, y, theta].

        Returns:
            float: sum of log probabilities of laser end points in the map.
        """

        angles: np.ndarray = self.lidar.get_ray_angles()
        distances: np.ndarray = self.lidar.get_sensor_values()

        x_world: np.ndarray = pose[0] + distances * np.cos(pose[2] + angles)
        y_world: np.ndarray = pose[1] + distances * np.sin(pose[2] + angles)

        x_grid, y_grid = self.grid.world2grid(x_world, y_world)

        mask_x: np.ndarray = (0 <= x_grid) & (x_grid < self.grid.size_x)
        mask_y: np.ndarray = (0 <= y_grid) & (y_grid < self.grid.size_y)

        mask: np.ndarray = mask_x & mask_y

        score: float = np.sum(self.grid.occupancy[x_grid[mask], y_grid[mask]])

        return score

    def get_corrected_pose(self,
        odom_pose: np.ndarray,
        odom_origin: np.ndarray = None
    ) -> np.ndarray:

        """
        Calculates the corrected pose in the map frame from the
        raw odometry pose and the odometry frame origin, provided
        as a second parameter or using the origin from the object.

        Args:
            odom_pose (np.ndarray): robot pose [x, y, theta] in odometry frame.
            odom_origin (np.ndarray, optional): pose [x, y, theta] of
            the odometry frame origin in map frame. Defaults to None.

        Returns:
            np.ndarray: corrected robot pose in the map frame.
        """

        if odom_origin is None:
            odom_origin = self.odom_origin

        angle: float = np.arctan2(odom_pose[1], odom_pose[0])
        distance: float = np.linalg.norm(odom_pose[:2])

        corrected_pose: np.ndarray = np.array([
            odom_origin[0] + distance * np.cos(odom_origin[2] + angle),
            odom_origin[1] + distance * np.sin(odom_origin[2] + angle),
            odom_origin[2] + odom_pose[2]
        ])

        return corrected_pose

    def localise(self, odom_pose: np.ndarray, max_iter: int = 150) -> float:
        """
        Calculates the robot's position in relation to the
        map and updates the origin of the odometry frame.

        Args:
            odom_pose (np.ndarray): robot pose [x, y, theta] in odometry frame.
            max_iter (int, optional): maximum number of iterations
            without improvement to exit the loop. Defaults to 150.

        Returns:
            float: score associated with the best location.
        """

        best_ref: np.ndarray = self.odom_origin
        best_score: float = self.score(
            self.get_corrected_pose(odom_pose))

        iter_counter: int = 0

        while iter_counter < max_iter:
            delta: np.ndarray = np.random.normal(0.0, [3.0, 3.0, 0.02])

            corrected: np.ndarray = self.get_corrected_pose(
                odom_pose, odom_origin=best_ref + delta)

            score: float = self.score(corrected)

            if score > best_score:
                best_ref += delta
                best_score = score

                iter_counter = 0

            else:
                iter_counter += 1

        self.odom_origin = best_ref

        return best_score

    def update_map(self, pose: np.ndarray) -> None:
        """
        Bayesian map update with new observation.

        Args:
            pose (np.ndarray): robot pose in format [x, y, theta].
        """

        angles: np.ndarray = self.lidar.get_ray_angles()
        distances: np.ndarray = self.lidar.get_sensor_values()

        cos: np.ndarray = np.cos(pose[2] + angles)
        sin: np.ndarray = np.sin(pose[2] + angles)

        x_free_space: np.ndarray = pose[0] + (distances - WALL_SPACING) * cos
        y_free_space: np.ndarray = pose[1] + (distances - WALL_SPACING) * sin

        x_wall_start: np.ndarray = pose[0] + (distances - WALL_WIDTH / 2) * cos
        y_wall_start: np.ndarray = pose[1] + (distances - WALL_WIDTH / 2) * sin

        x_wall_end: np.ndarray = pose[0] + (distances + WALL_WIDTH / 2) * cos
        y_wall_end: np.ndarray = pose[1] + (distances + WALL_WIDTH / 2) * sin

        for index in range(len(angles)):
            self.grid.increment_line(pose[0],
                                     pose[1],
                                     x_free_space[index],
                                     y_free_space[index],
                                     LOG_PROB_VACANT)

            self.grid.increment_line(x_wall_start[index],
                                     y_wall_start[index],
                                     x_wall_end[index],
                                     y_wall_end[index],
                                     LOG_PROB_OCCUPIED)

        self.grid.occupancy = np.clip(
            self.grid.occupancy, MIN_LOG_PROB, MAX_LOG_PROB)
