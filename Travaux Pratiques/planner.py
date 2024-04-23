import numpy as np

from occupancy_grid import OccupancyGrid


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

        shape = np.shape(self.grid.occupancy_map)
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            x = pos[0] + dx
            y = pos[1] + dy

            if x < 0 or x >= shape[0]:
                continue

            if y < 0 or y >= shape[1]:
                continue

            neighbors.append([x, y])

        return np.asarray(neighbors) if len(neighbors) > 0 else np.zeros((2, 0))

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """

        start = self.grid.conv_world_to_map(start[0], start[1])
        goal = self.grid.conv_world_to_map(goal[0], goal[1])
        
        shape = np.shape(self.grid.occupancy_map)
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

                if self.grid.occupancy_map[neigh[0], neigh[1]] > 2.0 or neigh in closed_set:
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
            path.append([*self.grid.conv_map_to_world(*curr), 0])
            curr = parent[curr[0], curr[1]].astype(int)

            if np.linalg.norm(parent[curr[0], curr[1]]) == 0.0:
                break

        return path[::-1]

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
