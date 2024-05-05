import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract

from place_bot.entities.lidar import LidarParams
from place_bot.entities.odometer import OdometerParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, Planner
from utils import Grid

from constants import *


# Definition of our robot controller
class MyRobot(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
        lidar_params: LidarParams = LidarParams(),
        odometer_params: OdometerParams = OdometerParams()
    ) -> None:

        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # Step counter to deal with init and display
        self.counter: int = 0

        self.grid: Grid = Grid(GRID_X_MIN,
                               GRID_Y_MIN,
                               GRID_X_MAX,
                               GRID_X_MAX,
                               GRID_RESOLUTION)

        self.tiny_slam = TinySlam(self.grid, self.lidar())
        self.planner = Planner(self.grid)

        self.pose: np.ndarray = np.array([0.0, 0.0, 0.0])

        self.waypoints_index = 0
        self.waypoints = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ]

    def control(self):
        """
        Main control function executed at each time step
        """

        score: float = self.tiny_slam.localise(self.odometer_values())
        self.pose = self.tiny_slam.get_corrected_pose(self.odometer_values())

        self.grid.display(self.pose)

        if self.counter == 0 or score > 50.0:
            self.tiny_slam.update_map(self.pose)

        self.path = self.planner.plan(self.pose, self.waypoints[self.waypoints_index])

        self.counter += 1

        goal = self.waypoints[self.waypoints_index]

        if np.linalg.norm(goal - self.pose) < 5.0:
            self.waypoints_index = (self.waypoints_index + 1) % len(self.waypoints)
            self.waypoints_index = int(self.waypoints_index)

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), self.pose, goal)

        return command
