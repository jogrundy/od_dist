"""
very simple test code for odds library.
"""

import os
import sys
import numpy as np
import unittest
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from odds import OD


class TestAllAlgos(unittest.TestCase):
    test_x = np.random.rand(10,2)
    test_x[5,:] = [4.1,4.2]
    test_y = [0,0,0,0,0,1,0,0,0,0]
    algo_lst = OD.algo_dict.keys()


    def test_algos(self):
        for algo in TestAllAlgos.algo_lst:
            od = OD(algo)
            out_scores = od.get_os(TestAllAlgos.test_x)
            out = np.argmax(out_scores)
            print(f'{algo}, max is {out}')
            for item in np.isnan(out_scores):
                self.assertEqual(item, False)

    def test_algos_norm(self):
        for algo in TestAllAlgos.algo_lst:
            od = OD(algo)
            out_scores = od.get_os(TestAllAlgos.test_x, norm=True)
            out = np.argmax(out_scores)
            print(f'{algo}, max is {out}')
            for item in np.isnan(out_scores):
                self.assertEqual(item, False)





if __name__ == '__main__':
    unittest.main()
