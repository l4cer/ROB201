import numpy as np


WALL_WIDTH = 2.0
WALL_SPACING = 20.0

MIN_LOG_PROB = -20.0
MAX_LOG_PROB =  20.0

LOG_PROB_FREE_SPACE = np.log10(0.05 / (1.0 - 0.05))
LOG_PROB_OCCUPIED = np.log10(0.95 / (1.0 - 0.95))
