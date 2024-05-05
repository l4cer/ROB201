import numpy as np


PI: float = np.pi
TAU: float = 2.0 * np.pi


#===================================================#
#                                                   #
#             Occupancy grid parameters             #
#                                                   #
#===================================================#

GRID_X_MIN: float = -800.0  # cm
GRID_Y_MIN: float = -800.0  # cm

GRID_X_MAX: float =  800.0  # cm
GRID_Y_MAX: float =  800.0  # cm

GRID_RESOLUTION: float = 2.0  # cm/pixel


#===================================================#
#                                                   #
#             Probabilistic lidar model             #
#                                                   #
#===================================================#

WALL_WIDTH: float = 2.0     # cm
WALL_SPACING: float = 20.0  # cm

MIN_LOG_PROB: float = -20.0
MAX_LOG_PROB: float =  20.0


def log_prob(prob: float) -> float:
    """
    Computes log(p/(1-p)) to map update with new observations.

    Args:
        prob (float): occupancy probability.

    Returns:
        float: occupancy log probability.
    """

    prob = np.clip(prob, 0.0, 1.0)

    return np.log10(prob / (1.0 - prob))


LOG_PROB_VACANT: float = log_prob(0.05)
LOG_PROB_OCCUPIED: float = log_prob(0.95)


#===================================================#
#                                                   #
#               Controller parameters               #
#                                                   #
#===================================================#

SCORE_MIN: float = 50.0

MIN_DIST_TO_EXPLORE: float = 200.0  # cm

LOOKAHEAD: int = 10

THRESHOLD: float = np.deg2rad(5)  # rad

KP_FORWARD: float = 0.02
KP_ROTATION: float = 0.5
