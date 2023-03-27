import numpy as np
from classes.lipschitz_env import Lipschitz_Environment

env = Lipschitz_Environment(lim=1.0, curve = 'cosine')
env.plot_curve()