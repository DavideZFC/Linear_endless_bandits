import matplotlib.pyplot as plt
import json
import numpy as np
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data

dir = 'results/_23_05_10-17_15_gaussianHT'

names = ['Chebishev', 'Fourier', 'Legendre', 'smooth_bins']

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