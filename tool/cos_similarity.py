#!/usr/bin/python -u

import sys
import numpy as np

for line in sys.stdin:
    s, t = line.rstrip('\n').split('\t')
    src = map(float, s.strip().split(' '))
    tgt = map(float, t.strip().split(' '))
    sim = np.sum((src/np.linalg.norm(src)) * (tgt/np.linalg.norm(tgt))) 
    print ("{:.4f}".format(sim))
