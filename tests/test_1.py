"""
very simple test code for odds library.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import numpy as np
from odds import OD
#list of algos supported by OD
algo_list = ['VAR', 'FRO', 'FRL', 'FRR', 'OCSVM', 'DBSCAN', 'GMM', 'IF', 'LSTM','GRU', 'AE', 'VAE', 'OP', 'GOP','RAND']
# very simple test vector, must be multidimensional to work
test_x = np.random.rand(10,2)
#seed pos 5 with outliers.
y = [0,0,0,0,0,1,0,0,0,0]
test_x[5,:] = [2.1,2.2]

for algo in algo_list:
    od = OD(algo)
    out_scores = od.get_os(test_x)
    out = np.argmax(out_scores)
    print(f'{algo} scores {out_scores}, max is {out}')

for algo in algo_list:
    od = OD(algo)
    out_scores = od.get_os(test_x, norm=True)
    out = np.argmax(out_scores)
    print(f'{algo} scores {out_scores}, max is {out}')
