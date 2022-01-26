#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt


with open('day20_image.txt') as infile:
    algo, image = infile.read().split('\n\n')

algo = [1 if char =='#' else 0 for char in algo]
image = np.array([[1 if char =='#' else 0 for char in row] for row in image.split('\n')])


print(len(algo))
algo[:5]


print(image.shape)
image[:5, :5]


plt.imshow(image)


padded = np.pad(image, 5, constant_values=new_image[0,0])


for i in range(2):
    new_image = np.zeros_like(padded[:-2, :-2])

    for r in range(1, padded.shape[0]-1):
        for c in range(1, padded.shape[1]-1):
            group = padded[r-1:r+2, c-1:c+2]
            idx = int(''.join([str(d) for d in group.ravel()]), 2)
            new_image[r-1, c-1] = algo[idx]

    padded = np.pad(new_image, 1, constant_values=new_image[0,0])


plt.imshow(new_image)


new_image.sum()


# # Part 2

padded = np.pad(image, 6, constant_values=new_image[0,0])


for i in range(50):
    new_image = np.zeros_like(padded[:-2, :-2])

    for r in range(1, padded.shape[0]-1):
        for c in range(1, padded.shape[1]-1):
            group = padded[r-1:r+2, c-1:c+2]
            idx = int(''.join([str(d) for d in group.ravel()]), 2)
            new_image[r-1, c-1] = algo[idx]

    padded = np.pad(new_image, 2, constant_values=new_image[0,0])
    #print(padded.shape)


plt.figure(figsize=(10,10))
plt.imshow(new_image)


new_image.sum()




