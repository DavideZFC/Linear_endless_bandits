import matplotlib.pyplot as plt
import json
import numpy as np
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data

dir = 'results/_23_05_10-17_15_gaussianHT'

names = ['smooth_bins']

for name in names:
    with open(dir+"/{}.json".format(name), "r") as f:
        # Convert the dictionary to a JSON string and write it to the file
        params = json.load(f)
    f.close()


    valm = 100000
    
    for key, val in params.items():
        if val<valm:
            valm = val
            best_key = key
    
    print(name+' '+best_key+' '+str(valm))


    d_list = [4, 5, 6, 8, 10, 12]
    m_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    bins_list = [5, 8, 10, 20]
    epsilon_list = [0.0, 0.01, 0.05, 0.1]

    mat = np.zeros((6,6,4,4))
    i = 0
    for d in d_list:
        j = 0
        for m in m_list:
            k = 0
            for b in bins_list:
                l = 0
                for e in epsilon_list:
                    key = 'd={} m={} b={} e={}'.format(d,m,b,e)
                    try:
                        mat[i,j,k,l] = params[key]
                    except:
                        pass
                    l += 1
                k += 1
            j += 1
        i += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(6)
y = np.arange(4)
z = np.arange(4)

minel = np.min(mat[mat != 0])
maxel = np.max(mat)

blu = np.array([0., 0., 1.])
red = np.array([1., 0.3, 0.3])

print(minel)
print(maxel)

for i in range(len(x)):
    for j in range(len(y)):
        for k in range(len(z)):
            m = np.mean(mat[i,:,j,k])
            if m > 0.5:
                v = mat[i,:,j,k]
                m = np.mean(v[v!=0])


                ratio = (m-minel)/(maxel-minel)
                color = (ratio*blu + (1-ratio)*red)
                ax.scatter(d_list[i], bins_list[j], epsilon_list[k], c=color, marker='o')

plt.savefig(dir+'/'+'heatmap.pdf')
plt.show()
