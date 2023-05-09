from classes.baselines.SmoothBins import SmoothBins
import numpy as np

N = 20
arms = np.linspace(-1,1,N)
learner = SmoothBins(arms, d=3, bins=5)
learner.reset()