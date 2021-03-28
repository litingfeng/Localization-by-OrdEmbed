"""
Created on 3/18/2021 2:11 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import numpy as np
from anchors import generate_anchors

def generate_boxes():
    anchors = generate_anchors(base_size=28, ratios=[0.5, 1.0, 2.0, 3.0])

if __name__ == '__main__':
    a = generate_anchors(base_size=28, ratios=[0.5, 1.0, 2.0, 3.0],
                         scales=np.array([1.0, 1., 2., 2.5]))
    print(a)
