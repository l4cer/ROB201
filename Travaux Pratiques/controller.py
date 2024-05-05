import cv2 as cv

from typing import Dict, List, Tuple, Union

from place_bot.entities.lidar import Lidar

from grid import Grid

from constants import *


def heuristic(curr: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """
    Calculates the cost between two points using some specific heuristic.
    In this case, the heuristic is precisely the Euclidean distance.

    Args:
        curr (Tuple[int, int]): curr position [x, y] in grid frame.
        goal (Tuple[int, int]): goal position [x, y] in grid frame.

    Returns:
        float: associated cost between these two points.
    """

    dx: float = float(curr[0] - goal[0])
    dy: float = float(curr[1] - goal[1])

    h: float = np.sqrt(dx**2 + dy**2)

    return h


def get_neighbors(grid: Grid, curr: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets valid neighbors of current node using 8-connexity.

    Args:
        grid (Grid): single instance of the Grid class.
        curr (Tuple[int, int]): curr position [x, y] in grid frame.

    Returns:
        List[Tuple[int, int]]: list of neighbors using 8-connexity.
    """

    neighbors: List[Tuple[int, int]] = []

    for dx, dy in [(-1,  1), ( 0,  1), ( 1,  1),
                   (-1,  0),           ( 1,  0),
                   (-1, -1), ( 0, -1), ( 1, -1)]:

        x: int = int(curr[0] + dx)
        y: int = int(curr[1] + dy)

        if x < 0 or grid.size_x <= x or y < 0 or grid.size_y <= y:
            continue

        neighbors.append((x, y))

    return neighbors


def erode_free_space(grid: Grid) -> np.ndarray:
    """
    Erodes free space to avoid robot collision during planning.

    Args:
        grid (Grid): single instance of the Grid class.

    Returns:
        np.ndarray: occupancy mask with free space eroded.
    """

    free_space: np.ndarray = grid.occupancy < 0.8 * MIN_LOG_PROB
    free_space = (255 * free_space).astype(np.uint8)

    kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (19, 19))

    free_space = cv.erode(free_space, kernel)

    return free_space > 0


def plan(grid: Grid, init: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    Computes a path using the A* algorithm,
    recomputes plan if init or goal changes.

    Args:
        grid (Grid): single instance of the Grid class.
        init (np.ndarray): init pose [x, y, theta] in world frame (theta unused).
        goal (np.ndarray): goal pose [x, y, theta] in world frame (theta unused).

    Returns:
        np.ndarray: path in world frame from init point to goal point.
    """

    init: Tuple[int, int] = grid.world2grid(init[0], init[1])
    goal: Tuple[int, int] = grid.world2grid(goal[0], goal[1])

    shape: Tuple[int, int] = (grid.size_x, grid.size_y)
    parent: np.ndarray = np.zeros((*shape, 2))

    g: np.ndarray = np.zeros(shape)
    f: np.ndarray = np.inf * np.ones(shape)
    f[init[0], init[1]] = heuristic(init, goal)

    open_set: set = set()
    open_set.add(init)

    closed_set: set = set()

    free_space: np.ndarray = erode_free_space(grid)

    while len(open_set) > 0:
        min_f: Union[None, float] = None
        curr: Union[None, Tuple[int, int]] = None

        for node in open_set:
            cost: float = f[node[0], node[1]]

            if min_f is None or cost < min_f:
                curr = node
                min_f = cost

        open_set.remove(curr)
        closed_set.add(curr)

        if heuristic(curr, goal) == 0.0:
            break

        for neigh in get_neighbors(grid, curr):
            if neigh in closed_set or not free_space[neigh[0], neigh[1]]:
                continue

            new_g: float = g[curr[0], curr[1]] + heuristic(curr, neigh)

            if new_g < g[neigh[0], neigh[1]] or neigh not in open_set:
                g[neigh[0], neigh[1]] = new_g
                f[neigh[0], neigh[1]] = new_g + heuristic(neigh, goal)

                parent[neigh[0], neigh[1]] = curr

                if neigh not in open_set:
                    open_set.add(neigh)

    curr: List[int, int] = goal
    path: List[List[float, float]] = []

    while heuristic(curr, init) > 0:
        path.append(grid.grid2world(curr[0], curr[1]))
        curr = parent[curr[0], curr[1]].astype(int)

        if np.linalg.norm(parent[curr[0], curr[1]]) == 0.0:
            break

    return np.asarray(path[::-1])


def separate_control(pose: np.ndarray, goal: np.ndarray) -> Dict[str, float]:
    """
    Controls actuators separately.

    Args:
        pose (np.ndarray): robot pose [x, y, theta] in map frame.
        goal (np.ndarray): goal position [x, y] in map frame.

    Returns:
        Dict[str, float]: dictionary with the respective actuator commands.
    """

    command: Dict[str, float] = {"forward": 0.0, "rotation": 0.0}

    delta: np.ndarray = goal - pose[:2]

    angle: float = np.arctan2(delta[1], delta[0]) - pose[2]
    distance: float = np.linalg.norm(delta)

    angle = angle % TAU
    angle = angle - TAU if angle > PI else angle

    if np.abs(angle) < THRESHOLD:
        command["forward"] = np.clip(KP_FORWARD * distance, -1.0, 1.0)
    else:
        command["rotation"] = np.clip(KP_ROTATION * angle, -1.0, 1.0)

    return command


class Controller:
    """Simple controller with path planning and path following"""

    def __init__(self, grid: Grid, lidar: Lidar) -> None:
        """
        Constructor of the Controller class.

        Args:
            grid (Grid): single instance of the Grid class.
            lidar (Lidar): single instance of the Lidar class.
        """

        self.grid: Grid = grid

        self.lidar: Lidar = lidar

        self.exploring: bool = True
        self.exploration_counter: int = 0

        self.goal: Union[None, np.ndarray] = None
        self.traj: Union[None, np.ndarray] = None

    def explore(self, pose: np.ndarray) -> None:
        """
        Frontier based exploration.

        Args:
            pose (np.ndarray): robot pose [x, y, theta] in map frame.
        """

        if self.goal is None or np.linalg.norm(pose[:2] - self.goal) < 10.0:
            free_space: np.ndarray = erode_free_space(self.grid)

            while True:
                x: int = np.random.randint(0, self.grid.size_x)
                y: int = np.random.randint(0, self.grid.size_y)

                if not free_space[x, y]:
                    continue

                if self.goal is not None:
                    self.exploration_counter += 1

                    print("Exploration counter: {}/{}".format(
                        self.exploration_counter, MAX_EXPLORATION_COUNTER))

                self.goal = np.array([*self.grid.grid2world(x, y)])
                break

            if self.exploration_counter >= MAX_EXPLORATION_COUNTER:
                self.exploring = False

                self.goal = np.array([0.0, 0.0])

            self.traj = plan(self.grid, pose, self.goal)

    def get_command(self, pose: np.ndarray) -> Dict[str, float]:
        """
        Calculates commands to control the robot.

        Args:
            pose (np.ndarray): robot pose [x, y, theta] in map frame.

        Returns:
            Dict[str, float]: dictionary with the respective actuator commands.
        """

        if self.exploring:
            self.explore(pose)

        delta: np.ndarray = self.traj - pose[:2]
        squared_norms: np.ndarray = np.einsum("ij,ij->i", delta, delta)

        index: int = np.clip(
            np.argmin(squared_norms) + LOOKAHEAD, 0, len(self.traj) - 1)

        command: Dict[str, float] = {"forward": 0.0, "rotation": 0.0}

        dist2origin: float = np.linalg.norm(pose[:2] - self.traj[index])

        if self.exploring or dist2origin > 5.0:
            command = separate_control(pose, self.traj[index])

        return command
