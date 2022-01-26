#!/usr/bin/env python
# coding: utf-8

import numpy as np


data_fname = 'day7_crabs.txt'
with open(data_fname, 'r') as datafile:
    positions = [int(l) for l in datafile.read().split(',')]


# # Part 1

positions = np.array(positions, dtype=int)


mean = np.median(positions)
mean


f1 = int(np.abs(positions - mean +1).sum())
f2 = int(np.abs(positions - mean).sum())
f3 = int(np.abs(positions - mean -1).sum())


min([f1, f2, f3])


f2


# # Part 2

print(positions.min())
print(positions.max())


# Each crab will have to end up somewhere between 0 and 1898 (let's say 2k)


0, 1, 1+2, 1+2+3, 1+2+3+4


for i in range(5):
    print((i+1)*i*.5)


max_size = 2_000
costs = np.zeros((len(positions), max_size), dtype=int)

for crab, p in enumerate(positions):
    i1 = np.arange(max_size-p)
    costs[crab, p:] = (i1+1) * i1 * .5
    
    i2 = np.arange(p, 0, -1)
    costs[crab, :p] = (i2+1) * i2 * .5


costs[1, :10]


costs.sum(axis=0).min()




