from typing import Dict

from place_bot.entities.robot_abstract import RobotAbstract

from place_bot.entities.lidar import LidarParams
from place_bot.entities.odometer import OdometerParams

from grid import Grid

from tiny_slam import TinySlam
from controller import Controller

from constants import *


class MyRobot(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
        lidar_params: LidarParams = LidarParams(),
        odometer_params: OdometerParams = OdometerParams()
    ) -> None:

        """
        Constructor of the MyRobot class.

        Args:
            lidar_params (LidarParams, optional): parameters that
            characterize the lidar sensor. Defaults to LidarParams().
            odometer_params (OdometerParams, optional): parameters to
            describe odometry process. Defaults to OdometerParams().
        """

        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        self.grid: Grid = Grid(GRID_X_MIN,
                               GRID_Y_MIN,
                               GRID_X_MAX,
                               GRID_X_MAX,
                               GRID_RESOLUTION)

        self.tiny_slam: TinySlam = TinySlam(self.grid, self.lidar())
        self.controller: Controller = Controller(self.grid, self.lidar())

        self.pose: np.ndarray = np.array([0.0, 0.0, 0.0])

        # Step counter to deal with init and display
        self.counter: int = 0

    def control(self) -> Dict[str, float]:
        """
        Main control function executed at each time step.

        Returns:
            Dict[str, float]: command for the robot's actuators
            in the format {"forward": float, "rotation": float}.
        """

        score: float = self.tiny_slam.localise(self.odometer_values())
        self.pose = self.tiny_slam.get_corrected_pose(self.odometer_values())

        command: Dict[str, float] = {"forward": 0.0, "rotation": 0.0}

        if self.counter < 50:
            self.tiny_slam.update_map(self.pose)

        else:
            if score > SCORE_MIN:
                self.tiny_slam.update_map(self.pose)

            command = self.controller.get_command(self.pose)

        self.grid.display(
            self.pose, goal=self.controller.goal, traj=self.controller.traj)

        self.counter += 1

        return command
