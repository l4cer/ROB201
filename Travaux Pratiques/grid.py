import cv2 as cv
import numpy as np

from typing import Tuple, Union


class Grid:
    """Simple occupancy grid object"""

    def __init__(self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        resolution: float
    ) -> None:

        """
        Constructor of the Grid class.

        Args:
            x_min (float): world minimum x coordinate in meters.
            y_min (float): world minimum y coordinate in meters.
            x_max (float): world maximum x coordinate in meters.
            y_max (float): world maximum y coordinate in meters.
            resolution (float): equivalent size of a pixel in meters.
        """

        self.x_min_world: float = x_min
        self.y_min_world: float = y_min

        self.x_max_world: float = x_max
        self.y_max_world: float = y_max

        self.resolution: float = resolution

        self.size_x: int = np.ceil((x_max - x_min) / resolution).astype(int)
        self.size_y: int = np.ceil((x_max - x_min) / resolution).astype(int)

        self.occupancy: np.ndarray = np.zeros((self.size_x, self.size_y))

    def world2grid(self,
        x_world: Union[float, np.ndarray],
        y_world: Union[float, np.ndarray]
    ) -> Union[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:

        """
        Converts from world coordinates to grid coordinates (i.e. cell indices).

        Args:
            x_world (float | np.ndarray): x coordinates in meters.
            y_world (float | np.ndarray): y coordinates in meters.

        Returns:
            Tuple[int, int] | Tuple[np.ndarray, np.ndarray]: tuple
            of x and y grid coordinates in cell numbers (~pixels).
        """

        x_grid: Union[float, np.ndarray] = (
            (x_world - self.x_min_world) / self.resolution)

        y_grid: Union[float, np.ndarray] = (
            (y_world - self.y_min_world) / self.resolution)

        if isinstance(x_grid, float):
            return int(x_grid), int(y_grid)

        if isinstance(x_grid, np.ndarray):
            return x_grid.astype(int), y_grid.astype(int)

        return x_grid, y_grid

    def grid2world(self,
        x_grid: Union[int, np.ndarray],
        y_grid: Union[int, np.ndarray]
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:

        """
        Converts from grid coordinates (i.e. cell indices) to world coordinates.

        Args:
            x_grid (int | np.ndarray): x coordinates in cell numbers (~pixels).
            y_grid (int | np.ndarray): y coordinates in cell numbers (~pixels).

        Returns:
            Tuple[float, float] | Tuple[np.ndarray, np.ndarray]:
            tuple of x and y world coordinates in meters.
        """

        x_world: Union[int, np.ndarray] = (
            self.x_min_world + x_grid * self.resolution)

        y_world: Union[int, np.ndarray] = (
            self.y_min_world + y_grid * self.resolution)

        if isinstance(x_world, np.ndarray):
            return x_world.astype(float), y_world.astype(float)

        return x_world, y_world

    def increment_value(self,
        x_world: Union[float, np.ndarray],
        y_world: Union[float, np.ndarray],
        value: float
    ) -> None:

        """
        Increments a value to an array of world points.

        Args:
            x_world (float | np.ndarray): x coordinates in meters.
            y_world (float | np.ndarray): y coordinates in meters.
            value (float): value to add to the cells of the points.
        """

        x_grid, y_grid = self.world2grid(x_world, y_world)

        mask_x: np.ndarray = (0 <= x_grid) & (x_grid < self.size_x)
        mask_y: np.ndarray = (0 <= y_grid) & (y_grid < self.size_y)

        mask: np.ndarray = mask_x & mask_y

        self.occupancy[x_grid[mask], y_grid[mask]] += value

    def increment_line(self,
        x0_world: float,
        y0_world: float,
        x1_world: float,
        y1_world: float,
        value: float
    ) -> None:

        """
        Increments a value to a line of points using Bresenham algorithm.

        Args:
            x0_world (float): x coordinate of starting point in meters.
            y0_world (float): y coordinate of starting point in meters.
            x1_world (float): x coordinate of ending point in meters.
            y1_world (float): y coordinate of ending point in meters.
            value (float): value to add to the cells of the points.
        """

        x0, y0 = self.world2grid(x0_world, y0_world)
        x1, y1 = self.world2grid(x1_world, y1_world)

        dx: int = x1 - x0
        dy: int = y1 - y0

        is_steep: bool = abs(dy) > abs(dx)

        if is_steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = y1 - y0

        error: int = int(dx / 2.0)
        step: int = 1 if 0 < dy else -1

        points: list = []

        y: int = y0
        for x in range(x0, x1 + 1):
            if is_steep:
                if 0 <= y and y < self.size_x and 0 <= x and x < self.size_y:
                    points.append([y, x])
            else:
                if 0 <= x and x < self.size_x and 0 <= y and y < self.size_y:
                    points.append([x, y])

            error -= abs(dy)
            if error < 0:
                y += step
                error += dx

        if len(points) > 0:
            points: np.ndarray = np.asarray(points)

            self.occupancy[points[:, 0], points[:, 1]] += value

    def display(self,
        pose: np.ndarray,
        goal: np.ndarray = None,
        traj: np.ndarray = None
    ) -> None:

        """
        Displays the robot's occupation grid and pose on the screen.

        Args:
            pose (np.ndarray): robot pose in format [x, y, theta].
            goal (np.ndarray, optional): goal point in world
            frame with coordinates in meters. Defaults to None.
            traj (np.ndarray, optional): trajectory points in world
            frame with coordinates in meters. Defaults to None.
        """

        img: np.ndarray = cv.flip(self.occupancy.T, 0)

        min_val: float = np.min(img)
        max_val: float = np.max(img)

        img = (255 * (img - min_val) / (max_val - min_val)).astype(np.uint8)
        img = cv.applyColorMap(src=img, colormap=cv.COLORMAP_JET)

        if traj is not None:
            traj_x, traj_y = self.world2grid(traj[:, 0], -traj[:, 1])
            points: np.ndarray = np.hstack(
                (traj_x[:, np.newaxis], traj_y[:, np.newaxis]))

            for i in range(len(traj_x) - 1):
                cv.line(img, points[i], points[i+1], (180, 180, 180), 2)

        if goal is not None:
            point: np.ndarray = self.world2grid(goal[0], -goal[1])
            cv.circle(img, point, 3, (255, 255, 255), -1)

        point1: np.ndarray = self.world2grid(pose[0], -pose[1])
        point2: np.ndarray = self.world2grid(
             pose[0] + 20.0 * np.cos(pose[2]),
            -pose[1] - 20.0 * np.sin(pose[2]))

        cv.arrowedLine(img, point1, point2, (0, 0, 255), thickness=2)
        cv.imshow("SLAM map", img)
        cv.waitKey(1)
