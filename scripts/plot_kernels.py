import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.linspace(-1,1,N)

fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=1.0)

def fun(x,n):
    return 1+(np.sin(x*n/2))/(np.sin(x/2))*np.cos((n+1)*(x)/2)

axs[0, 0].plot(x, fun(x,5))
axs[0, 0].set_title('n=5')
axs[0, 1].plot(x, fun(x,10))
axs[0, 1].set_title('n=10')
axs[1, 0].plot(x, fun(x,20))
axs[1, 0].set_title('n=20')
axs[1, 1].plot(x, fun(x,40))
axs[1, 1].set_title('n=40')

fig.savefig('kernels.pdf')
plt.show()