""" A set of robotics control functions """

import random
import numpy as np

from scipy.signal import convolve

from utils import OccupancyGrid

KP_SPEED = 0.001
KP_ROTATION = 0.2

FOV = np.deg2rad(240)
KERNEL = np.ones(61)

prev_target = 0.0


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def compute_h(self, curr, goal):
        dx = curr[0] - goal[0]
        dy = curr[1] - goal[1]

        return np.sqrt(dx**2 + dy**2)

    def get_neighbors(self, pos):
        neighbors = []

        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            x = pos[0] + dx
            y = pos[1] + dy

            if x < 0 or x >= self.grid.size_x:
                continue

            if y < 0 or y >= self.grid.size_y:
                continue

            neighbors.append([x, y])

        return np.asarray(neighbors) if len(neighbors) > 0 else np.zeros((2, 0))

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """

        start = self.grid.world2grid(start[0], start[1])
        goal = self.grid.world2grid(goal[0], goal[1])
        
        shape = np.shape(self.grid.occupancy)
        parent = np.zeros((shape[0], shape[1], 2))

        g = np.zeros(shape)
        f = np.inf * np.ones(shape)
        f[start[0], start[1]] = self.compute_h(start, goal)

        open_set = set()
        closed_set = set()
        open_set.add(start)

        while len(open_set) > 0:
            #print(len(open_set), len(closed_set))
            min_f = None
            curr = None

            for node in open_set:
                cost = f[node[0], node[1]]

                if min_f is None or cost < min_f:
                    curr = node
                    min_f = cost

            open_set.remove(curr)
            closed_set.add(curr)

            if self.compute_h(curr, goal) == 0.0:
                break

            for neigh in self.get_neighbors(curr):
                neigh = (int(neigh[0]), int(neigh[1]))

                if self.grid.occupancy[neigh[0], neigh[1]] > 2.0 or neigh in closed_set:
                    continue
                
                new_g = g[curr[0], curr[1]] + self.compute_h(curr, neigh)

                if new_g < g[neigh[0], neigh[1]] or neigh not in open_set:
                    g[neigh[0], neigh[1]] = new_g
                    f[neigh[0], neigh[1]] = new_g + self.compute_h(neigh, goal)
                    parent[neigh[0], neigh[1]] = curr

                    if neigh not in open_set:
                        open_set.add(neigh)

        path = []
        curr = goal

        while self.compute_h(curr, start) > 0:
            path.append([*self.grid.grid2world(*curr), 0])
            curr = parent[curr[0], curr[1]].astype(int)

            if np.linalg.norm(parent[curr[0], curr[1]]) == 0.0:
                break

        return path[::-1]

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal



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

    K = 10.0
    D_SAFE = 0.5

    angles = lidar.get_ray_angles()
    distances = lidar.get_sensor_values()

    delta = goal_pose[:2] - current_pose[:2]
    norm = np.linalg.norm(delta)

    grad = K * delta / norm

    index = np.argmin(distances)
    angle_min, dist_min = angles[index], distances[index]

    target = dist_min * np.array([np.cos(angle_min), np.sin(angle_min)])

    grad += (K / dist_min**3) * (1.0 / dist_min - 1.0 / D_SAFE) * target

    error = np.arctan2(grad[1], grad[0]) - current_pose[2]

    if np.abs(error) < 0.01:
        speed = np.clip(0.03 * np.linalg.norm(grad), -1.0, 1.0)
        rotation = 0.0

    else:
        speed = 0.0
        rotation = np.clip(0.2 * error, -1.0, 1.0)

    return {"forward": speed, "rotation": rotation}
