import matplotlib.pyplot as plt
import json
import numpy as np
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data

dir = 'results\_23_04_17-15_20_gaussian'

with open(dir+"/running_times.json", "r") as f:
    # Convert the dictionary to a JSON string and write it to the file
    running_times = json.load(f)

# makes the barplot
plt.bar(running_times.keys(), np.log(np.array(list(running_times.values()))))

plt.xticks(rotation=15)
plt.tick_params(axis='x', labelsize=8)
plt.ylabel('log(time)')
plt.savefig(dir+'/running_times.pdf')
plt.show()


labels = list(running_times.keys())
print(labels)

c = 0
for l in labels:
    results = np.load(dir+'/'+l+'.npy')
    
    # make nonparametric confidence intervals
    low, high = bootstrap_ci(results, resamples=1000)
    T = len(low)

    # make plot
    plot_data(np.arange(0,T), low, high, col='C{}'.format(c), label=l)

    # update color
    c += 1

    print(l+' done')


plt.legend()
plt.title('Regret curves')
plt.savefig(dir+'/regret_plot.pdf')



