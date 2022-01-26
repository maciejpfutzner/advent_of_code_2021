#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np


def load_steps(fname='day22_cubes.txt'):
    with open(fname) as infile:
        data_raw = [l.strip() for l in infile]

    pattern = r'(\w+) x=(.*)\.\.(.*),y=(.*)\.\.(.*),z=(.*)\.\.(.*)'
    steps = []
    for line in data_raw:
        state, xmin, xmax, ymin, ymax, zmin, zmax = re.match(pattern, line).groups()
        state = 1 if state == 'on' else 0
        steps.append((state, (int(xmin), int(xmax)+1),
                      (int(ymin), int(ymax)+1), (int(zmin), int(zmax)+1)))
    return steps


steps = load_steps()


steps[:25]


# # Part 1

steps = load_steps()
#steps = load_steps('day22_example.txt')
#steps=[(1, (10, 10), (10, 10), (10, 10))]


region = np.zeros((101, 101, 101))
zero = (50, 50, 50)
step_counters = []

for state, xrange, yrange, zrange in steps:
    xi, xf = xrange[0] + zero[0], xrange[1] + zero[0]
    yi, yf = yrange[0] + zero[1], yrange[1] + zero[1]
    zi, zf = zrange[0] + zero[2], zrange[1] + zero[2]
    if (0<= xi <= region.shape[0] and 0<= xf <= region.shape[0] and
        0<= yi <= region.shape[1] and 0<= yf <= region.shape[1] and
        0<= zi <= region.shape[2] and 0<= zf <= region.shape[2]):
        print(xi, xf, yi, yf, zi, zf)
        region[xi:xf, yi:yf, zi:zf] = state
    step_counters.append(region.sum())


int(region.sum())


# # Part 2

arr = np.array([(xlo, xhi, ylo, yhi, zlo, zhi)
                for _, (xlo, xhi), (ylo, yhi), (zlo, zhi) in steps])
print(np.min(arr, axis=0)[::2])
print(np.max(arr, axis=0)[1::2])


# ### Not enough memory for previous approach

# ### We gotta keep track of the abstract cuboids
# - Easy to figure out sizes of cuboids
# - Tricky to figure out the overlaps
# - Then applying all overlaps in correct order
# 
# How to do it
# - Keep a number of active cubes
# - Keep a list of cuboid regions that are on
# - Whenever adding a new one that's on, check for overlap with existing on-regions
#     - If there's overlap, subtract volume from the counter
# - Then check for overlap with tracked off-regions (only part of on-cuboids)
#     - If there's overlap, add new cuboid that's on
#         - Add its volume to the counter
# - When adding a new one that's off, check for overlap with all existing on-regions
#     - If there's no overlap, ignore it (unlikely)
#     - If there's overlap, create a new cuboid to track it (state = off)
#         - Attach those to the mother on-cuboid
#         - ~~Order is important~~
#     - Check all newly created cuboids for overlaps
#         - Subtract (volume of created cuboids - volume of their overlaps) from counter

class Cuboid():
    def __init__(self, xrange, yrange, zrange):
        self.xlo, self.xhi = xrange
        self.ylo, self.yhi = yrange
        self.zlo, self.zhi = zrange
        self.offs = []
    
    def __eq__(self, other):
        eq = (self.xlo == other.xlo and
              self.xhi == other.xhi and
              self.ylo == other.ylo and
              self.yhi == other.yhi and
              self.zlo == other.zlo and
              self.zhi == other.zhi)
        return eq
    
    @property
    def ranges(self):
        return ((self.xlo, self.xhi), (self.ylo, self.yhi),
                (self.zlo, self.zhi))
    
    @property
    def volume(self):
        v = ((self.xhi - self.xlo) *
             (self.yhi - self.ylo) *
             (self.zhi - self.zlo))
        assert v > 0
        return v
    
    @property
    def vertices(self):
        vertices = [(x,y,x)
                    for x in (self.xlo, self.xhi)
                    for y in (self.ylo, self.yhi)
                    for z in (self.zlo, self.zhi)]
        return vertices
    
    def overlap(self, other):
        xlo = max(self.xlo, other.xlo)
        ylo = max(self.ylo, other.ylo)
        zlo = max(self.zlo, other.zlo)
        
        xhi = min(self.xhi, other.xhi)
        yhi = min(self.yhi, other.yhi)
        zhi = min(self.zhi, other.zhi)
        
        if xlo < xhi and ylo < yhi and zlo < zhi:
            return Cuboid((xlo, xhi), (ylo, yhi), (zlo, zhi))
        else:
            return False
    
    def is_inside(self, point):
        x,y,z = point
        if (self.xlo < x < self.xhi and 
            self.ylo < y < self.yhi and 
            self.zlo < z < self.zhi):
            return True
        else:
            return False
        
    def __repr__(self):
        return (f'Cuboid(({self.xlo}, {self.xhi}), '
                f'({self.ylo}, {self.yhi}), '
                f'({self.zlo}, {self.zhi}))')
    
    def __hash__(self):
        return hash(self.ranges)


volumes = []

for i, (state, *ranges) in enumerate(steps):
    new = Cuboid(*ranges)
    for sign, other in volumes.copy():
        overlap = new.overlap(other)
        if overlap:
            # Two "positive" volumes overlapping create a negative volume
            # but a positive volume overlapping with a negative one
            # should add back a positive number
            volumes.append((-sign, overlap))
    if state:
        volumes.append((1, new))

volume = sum([c.volume*sign for sign, c in volumes])
print(volume)


raise RuntimeError


# ### Original solution - The overlaps are too complicated to keep track of (I thought)
# Instead, whenever there's an overlap, let's try to split one of the cuboitds into a few smaller ones. Then remove the overlapping part altogether.
# 
# Algorithm
# - When adding a new on-cube, look for overlaps with existing on-cubes
# - For each overlap
#     - Find the cube that has at least one vertex inside the other (these vertices should be shared with the overlap cuboid)
#     - Take one such vertex and the coordinates of the outer cube
#     - Create 8 new volumes, using one of the vertex coordinates and one of the eight vertices of the outer cube
#         - Re-sort low and high coordinates to feed numbers into Cuboid class
#     - Remove the original outer volume and cuboid that's identical with the overlap (if exists)
#     - See if the inner cube overlaps with any of the new cuboids. If so, repeat the process (recursively?)
# - When adding an off-cube, do the same, except don't keep any of the non-overlapping off volumes

def split_both(c1, c2):
    new_ranges = []
    for i in range(3):
        lo1, hi1 = c1.ranges[i]
        lo2, hi2 = c2.ranges[i]
        ll = sorted(set([lo1, lo2, hi1, hi2]))
        new_r = []
        for i in range(len(ll)-1):
            new_r.append([ll[i], ll[i+1]])
        new_ranges.append(new_r)
    new_ranges

    c1_children, c2_children = [], []
    for xrange in new_ranges[0]:
        for yrange in new_ranges[1]:
            for zrange in new_ranges[2]:
                new = Cuboid(xrange, yrange, zrange)
                if new.overlap(c1):
                    c1_children.append(new)
                elif new.overlap(c2):
                    c2_children.append(new)

    return c1_children, c2_children


get_ipython().run_cell_magic('timeit', '', 'split_both(c1, c2)')


#steps = load_steps('day22_overlap_example.txt')
steps = load_steps()[:20]


on_cuboids = []

for i, (state, *ranges) in enumerate(steps):
    print(f'Step {i}, {len(on_cuboids)} tracked cuboids')
    
    master_new = Cuboid(*ranges)
    new_cuboids = [master_new]
    non_overlapping = set()
    
    repeat = True
    while repeat:
        repeat = False
        for other in set(on_cuboids) - non_overlapping:
            for new in new_cuboids:
                overlap = new.overlap(other)
                if overlap:
                    news, others = split_both(new, other)
                    
                    #print(f'Attempting to remove {other}')
                    on_cuboids.remove(other)
                    on_cuboids.extend(others)
                    
                    new_cuboids.remove(new)
                    new_cuboids.extend(news)
                    
                    repeat = True
                    break
            if repeat:
                break
            non_overlapping.add(other)
    if state:
        # After everything is said and done, add all the new
        # on cuboids to the global list
        on_cuboids.extend(new_cuboids)

total_volume = sum([c.volume for c in on_cuboids])
print(total_volume)


# Try to speed it up by using a hierarchy of volumes and storing the children below the master volume...

on_cuboids = {}

for i, (state, *ranges) in enumerate(steps):
    print(f'Step {i}, {len(on_cuboids)} tracked cuboids')
    
    master_new = Cuboid(*ranges)
    new_cuboids = [master_new]
    #non_overlapping = set()
    
    repeat = True
    while repeat:
        repeat = False
        for new in new_cuboids:
            #for other in set(on_cuboids) - non_overlapping:
            for master_other in on_cuboids:
                if new.overlap(master_other):
                    for other in on_cuboids[master_other]:
                        overlap = new.overlap(other)
                        if overlap:
                            news, others = split_both(new, other)

                            on_cuboids[master_other].remove(other)
                            on_cuboids[master_other].extend(others)

                            new_cuboids.remove(new)
                            new_cuboids.extend(news)

                            repeat = True
                            break
                    if repeat:
                        break
            if repeat:
                break
            #non_overlapping.add(other)
    if state:
        # After everything is said and done, add all the new
        # on cuboids to the global list
        on_cuboids[master_new] = new_cuboids

total_volume = sum([c.volume for c in on_cuboids])
print(total_volume)


# ### Old, overcomplicated (and wrong) code

counter = 0
on_cuboids = []
overlaps = []

for i, (state, *ranges) in enumerate(steps[:20]):
    if state:
        print('New on-region')
        new = Cuboid(*ranges)
        print(new)
        new_cuboids = [new]
        counter += new.volume
        print(f'Adding {new.volume} to the counter')
        
        for c_on in overlaps:
            # Check overlap with other, pre-registered overlaps
            overlap = new.overlap(c_on)
            if overlap:
                # Add back to the counter cause we'll remove it in next step
                print('Found overlap with another overlap, '
                      f'adding {overlap.volume} to counter')
                counter += overlap.volume
            
        for c_on in on_cuboids:
            # Check overlap with other on-cuboids
            overlap = new.overlap(c_on)
            if overlap:
                print('Found overlap with another on-cube, '
                      f'subtracting {overlap.volume} from the counter')
                counter -= overlap.volume
                overlaps.append(overlap)
            for c_off in c_on.offs:
                # Check overlap with off-areas of on-cuboids
                overlap2 = new.overlap(c_off)
                if overlap2:
                    print('Found overlap with an off-region, '
                          f'adding {overlap2.volume} back to the counter')
                    # If there's overlap with an off-area,
                    # Create a new independend on-cuboid
                    counter += overlap2.volume
                    new_cuboids.append(overlap2)
                    print('And creating new cuboid', overlap2)
        on_cuboids.extend(new_cuboids)
    else:
        print('New off-region')
        new_off = Cuboid(*ranges)
        print(new_off)
        new_offs = []
        for c_on in on_cuboids:
            overlap = new_off.overlap(c_on)
            if overlap:
                print('Found overlap with an on-cube, creating sub-region')
                c_on.offs.append(overlap)
                new_offs.append(overlap)
                
        print('\nChecking for overlaps with other off-regions')
        found_pairs = set()
        for i, off1 in enumerate(new_offs):
            print()
            counter -= off1.volume
            print(f'Subtracting {off1.volume} from the counter')
            for c_on in on_cuboids:
                print('Looking at off-regions added to', c_on)
                for off2 in c_on.offs:
                    print('Found off-region', off2)
                    if off1 != off2:
                        overlap = off1.overlap(off2)
                        if overlap:
                            if ((off1, off2) not in found_pairs and 
                                (off2, off1) not in found_pairs):
                                print('Found overlap with another off-region, '
                                     f'adding {overlap.volume} back to the counter')
                                counter += overlap.volume
                                found_pairs.add((off2, off1))
                                
    print(f'\nCube counter: {counter}')
    print(f'But should be {int(step_counters[i])}')
    if diff := counter - step_counters[i]:
        print(f'That\'s a difference of {diff}')
    print()
    


7669371029


988224.0


counter

