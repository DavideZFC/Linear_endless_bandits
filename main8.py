from classes.curve_fit import curve_fit
from classes.Logli import Logli
import matplotlib.pyplot as plt

curve = 'gaussian'
env = curve_fit(curve=curve)
T = 10000

agent = Logli(env, T)
agent.execute()

plt.plot(agent.reward_story)
plt.show()