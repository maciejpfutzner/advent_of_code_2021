#!/usr/bin/env python
# coding: utf-8

data_filename = 'day13_dots.txt'
with open(data_filename) as datafile:
    dots = [l.strip() for l in datafile.readlines()]

dots = [coords.split(',') for coords in dots]
dots = [tuple(int(c) for c in dot) for dot in dots]


data_filename2 = 'day13_folding.txt'
with open(data_filename2) as datafile:
    instructions = [l.strip() for l in datafile.readlines()]


instructions


# # Part 1

import numpy as np
from matplotlib import pyplot as plt


dots = np.array(dots)
dots[:10]


# Do this automatically
paper = np.zeros((1311, 895)).astype(bool)
paper[list(zip(*dots))] = True


plt.figure(figsize=(20,20))
plt.imshow(paper, cmap='gray_r')


# ### Wrong coordinates!

# First fold
print(instructions[0])


x = 655

fold = paper[x+1:, :]
paper = paper[:x, :]


paper.shape


fold.shape


paper |= np.flip(fold, axis=0)


paper.sum()


# # Part 2

foldings = [i.split(' ')[-1].split('=') for i in instructions]


# Do this automatically
paper = np.zeros((1311, 895)).astype(bool)
paper[list(zip(*dots))] = True

for axis, val in foldings:
    val = int(val)
    if axis == 'x':
        fold = paper[val+1:, :]
        paper = paper[:val, :]
        paper |= np.flip(fold, axis=0)
    else:
        fold = paper[:, val+1:]
        paper = paper[:, :val]
        paper |= np.flip(fold, axis=1)


plt.imshow(paper)


plt.imshow(paper.T)




