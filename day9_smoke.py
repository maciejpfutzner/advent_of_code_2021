#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt


data_filename = 'day9_heightmap.txt'
with open(data_filename) as datafile:
    data = [l.strip() for l in datafile.readlines()]


heights = np.array([[int(h) for h in row] for row in data])


heights.shape


# # Part 1

plt.imshow(heights)


def xy_in_bounds(x, y, heights=heights):
    if x < 0 or x >= heights.shape[0]:
        return False
    elif y < 0 or y >= heights.shape[1]:
        return False
    else:
        return True


def find_neighbours(x, y, heights=heights):
    neighbours = np.array([(x-1, y), (x+1, y),
                           (x, y-1), (x, y+1)])
    neighbours = [n for n in neighbours if xy_in_bounds(*n, heights)]
    return neighbours


def is_low(x, y, heights=heights):
    neighbours = find_neighbours(x, y, heights)
    if (np.array([heights[x,y] for x,y in neighbours]) >
        heights[x,y]).all():
        return True
    else:
        return False


is_low(5,5)


lows = np.zeros_like(heights).astype(bool)
for i in range(100):
    for j in range(100):
        lows[i,j] = is_low(i, j)


lows.sum()


(heights[lows] + 1).sum()


# # Part 2

plt.figure(figsize=(10,10))
plt.imshow(heights)


plt.figure(figsize=(10,10))
plt.imshow(heights==9)


def find_neighbours_unexplored(x, y, explored, heights=heights):
    neighbours = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    neighbours = [n for n in neighbours if xy_in_bounds(*n, heights)]
    neighbours = [n for n in neighbours if not explored[n]]
    return neighbours


sample = heights[5:15, 10:20]
s_lows = lows[5:15, 10:20]
plt.imshow(sample)


plt.imshow(s_lows)


# ### Setup

low_locs = list(zip(*np.where(s_lows)))

# Array to store places already checked (or 9)
explored = sample==9

basin_size = 1
loc = low_locs[0]
explored[loc] = True
to_check = []


plt.imshow(explored)
plt.scatter(loc[1], loc[0])


# ### Single step

def step(loc, basin_size):
    neighbours = find_neighbours_unexplored(*loc, explored, sample)
    print(neighbours)

    basin_size += len(neighbours)
    for n in neighbours:
        explored[n] = True
    to_check.extend(neighbours)
    
    #plt.imshow(explored)
    #plt.scatter(loc[1], loc[0])


while len(to_check) > 0:
    loc = to_check.pop()
    step(loc)


to_check


basin_size


# ### Now seriously

def step(loc, basin_size, explored, to_check):
    neighbours = find_neighbours_unexplored(*loc, explored, heights)

    basin_size += len(neighbours)
    for n in neighbours:
        explored[n] = True
    to_check.extend(neighbours)
    
    return basin_size, explored, to_check


low_locs = list(zip(*np.where(lows)))


# Array to store places already checked (or 9)
explored = heights == 9

all_sizes = []
for loc in low_locs:
    basin_size = 1
    explored[loc] = True
    to_check = [loc]

    while len(to_check) > 0:
        loc = to_check.pop()
        basin_size, explored, to_check = step(loc, basin_size, explored, to_check)
        
    all_sizes.append(basin_size)


t1, t2, t3 = np.sort(all_sizes)[-3:]
t1, t2, t3


t1*t2*t3


# ### Try a BFS/floodfill approach

from collections import deque


get_ipython().run_cell_magic('timeit', '', 'explored = set()\n#explored = heights == 9 # minimally faster with array lookup\nsizes = []\n\nfor row, col in low_locs:\n    q = deque()\n    size = 0\n    if (row, col) not in explored:\n    #if not explored[row, col]:\n        q.append((row,col))\n    while q:\n        row, col = q.popleft()\n        if (row, col) in explored:\n        #if explored[row, col]:\n            continue\n        else:\n            explored.add((row, col))\n            #explored[row, col] = True\n            size += 1\n            # add neighbours to queue\n            for dd in [(0,1), (1,0), (0,-1), (-1,0)]:\n                rr = row + dd[0]\n                cc = col + dd[1]\n                if (0 <= rr < heights.shape[0] and\n                    0 <= cc < heights.shape[1] and\n                    heights[rr, cc] != 9):\n                    q.append((rr, cc))\n    sizes.append(size)')


t1, t2, t3 = np.sort(all_sizes)[-3:]
print(t1, t2, t3)
t1*t2*t3


# # Make the animation

from PIL import Image
import imageio


# Array to store places already checked (or 9)
explored = heights == 9
#fig, ax = plt.subplots(figsize=(10,10))
filenames = []
frame_counter = 0

all_sizes = []
for loc in low_locs:
    basin_size = 1
    explored[loc] = True
    to_check = [loc]

    while len(to_check) > 0:
        loc = to_check.pop()
        basin_size, explored, to_check = step(loc, basin_size, explored, to_check)
        
        # Save animation frame
        if frame_counter > 0:
            filename = f'animation/{frame_counter}.png'
            filenames.append(filename)
            #ax.imshow(explored)
            #fig.savefig(filename)
            img_array = np.kron(~explored, np.ones([4,4]))
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            img.save(filename)
        frame_counter += 1
        
    all_sizes.append(basin_size)


#with imageio.get_writer('animation_day9.gif', mode='I', duration=) as writer:
images = []
for filename in filenames[::5]:
    image = imageio.imread(filename)
    #writer.append_data(image)
    images.append(image)
imageio.mimsave('animation_day9_sub.gif', images, fps=55)




