import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_data(x, v1, v2, col, label):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, (v1+v2)/2, label=label, color=col)
    ax.fill_between(x, y1=v1, y2=v2, color=col, alpha=0.3)

    # plt.plot(x, (v1+v2)/2, label=label, color=col)
    # plt.fill_between(x, y1=v1, y2=v2, color=col, alpha=0.3)