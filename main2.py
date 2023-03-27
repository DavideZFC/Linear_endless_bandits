import numpy as np
from classes.lipschitz_env import Lipschitz_Environment

env = Lipschitz_Environment(curve = 'random')
env.plot_curve()