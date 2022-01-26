#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
from collections import defaultdict


def load_scans(filename='day19_scanners.txt'):
    with open(filename) as infile:
        scanners = infile.read().split('\n\n')

    scans = {}
    for scanner_data in scanners:
        scanner_data = scanner_data.split('\n')
        n_scanner = int(re.findall('\d+', scanner_data[0])[0])
        data = np.array([[int(c) for c in r.split(',')]
                         for r in scanner_data[1:]])
        scans[n_scanner] = data
    scans = [scans[i] for i in range(len(scans))]
    return scans


scans = load_scans()
ex_scans = load_scans('day19_example.txt')


scans = ex_scans


# # Part 1

for data in scans:
    print(data.shape)
    print(data.min(axis=0))
    print(data.max(axis=0))
    print()


# ### Find all coordinate transformations

# Must be a nicer way to do this
units = []
for i in range(3):
    tmp = np.array([0, 0, 0])
    tmp[i] += 1
    units.append(tmp)
units


rotations = []
for xdir in range(3):
    for xsign in [1, -1]:
        for ydir in set(range(3)) - {xdir}:
            if ydir != xdir:
                for ysign in [1, -1]:
                    zdir = (set(range(3)) - {xdir, ydir}).pop()
                    if str(xdir)+str(ydir) in '0120':
                        zsign = xsign*ysign
                    else:
                        zsign = -xsign*ysign 
                    x = units[xdir] * xsign
                    y = units[ydir] * ysign
                    z = units[zdir] * zsign
                    rot = np.array([x,y,z])
                    rotations.append(rot)
                    #print(rot)
len({tuple(tuple(c) for c in r) for r in orientations})


vecs = [
    [-1,-1,1],
    [-2,-2,2],
    [-3,-3,3],
    [-2,-3,1],
    [5,6,-4],
    [8,0,7]
]

#for rot in rotations:
#    for vec in vecs:
#        print(rot.dot(vec))
#    print()


# ### Try brute force approach

s1 = 0
s2 = 1

data1 = scans[s1]
data2 = scans[s2]


set1 = {*(tuple(c for c in r) for r in data1)}
found_match = False

#for rot in rotations:
#    rotated = rot.dot(data2.T).T
#    mean_diff = (data1.mean(axis=0) - rotated.mean(axis=0)).astype(int)
#    for i in range(-100, 100):
#        for j in range(-100, 100):
#            for k in range(-100, 100):
#                offset = mean_diff + np.array([i, j, k])
#                set2 = {*(tuple(c for c in r) for r in rotated + offset)}
#                if len(set1 & set2) > 0:
#                    found_match = True
#                    print('found match')
#                    break
#            if found_match:
#                break
#        if found_match:
#            break


i,j,k





# ### That's too slow

data1 = ex_scans[0]
data2 = ex_scans[1]


offset = np.array([68,-1246,-43])


set1 = {*(tuple(c for c in r) for r in data1)}
for rot in rotations:
    rotated = rot.dot((data2).T).T
    set2 = {*(tuple(c for c in r) for r in rotated + offset)}
    if len(set1 & set2) > 0:
        found_match = True
        print('found match')
        break


# ### Look for an invariant - distances to other beacons

def get_dists(data):
    dists = []
    for row in data:
        dd = {*(tuple(c for c in r) for r in data - row)}
        dists.append(dd - {(0,0,0)})
    return dists


overlaps = {}
for s1, data1 in enumerate(scans):
    for s2, data2 in list(enumerate(scans))[s1+1:]:
        #print(f'checking scanners {s1} and {s2}')
        dists1 = get_dists(data1)
        for rot in rotations:
            rotated = rot.dot((data2).T).T
            dists2 = get_dists(rotated)
            
            matches = []
            for i, dd1 in enumerate(dists1):
                for j, dd2 in enumerate(dists2):
                    if (overlap := len(dd1 & dd2)) > 10:
                        matches.append((i, j, overlap))
            if len(matches) >= 12:
                print(f"Overlap between scanners {s1} and {s2}")
                overlaps[(s1, s2)] = (matches, rot)


# Not the correct way to get all beacons
sum([len(ss) for ss in scans]) - sum([len(v) for v in overlaps.values()])


def get_correction(s1, s2, overlaps, scans=scans):
    reverse = False
    if s1 > s2:
        s1, s2 = s2, s1
        reverse = True
        
    if (s1, s2) not in overlaps:
        print('Not an overlapping pair')
        return None
    
    matches, rot = overlaps[(s1, s2)]
    bid1, bid2, _ = matches[0]
    b1 = scans[s1][bid1]
    b2 = scans[s2][bid2]
    offset = b1 - rot.dot(b2)
    
    if reverse:
        rot = np.linalg.inv(rot)
        offset = b2 - rot.dot(b1)
    
    rot = rot.copy()
    offset = offset.copy()
    def correction(data):
        corrected = rot.dot((data).T).T + offset
        return corrected
    
    return correction


#correction14_0 = get_correction(14, 0, overlaps)
#set14_0 = {*(tuple(c for c in r) for r in correction14_0(scans[0]))}
#set14_0 & {*(tuple(c for c in r) for r in scans[14])}


#correction0_14 = get_correction(0, 14, overlaps)
#set0_14 = {*(tuple(c for c in r) for r in correction0_14(scans[14]))}
#set0_14 & {*(tuple(c for c in r) for r in scans[0])}


corrections[19][4](corrections[4][19](np.eye(3)))


corrections = defaultdict(dict)
for s1, s2 in overlaps.keys():
    corrections[s1][s2] = get_correction(s1, s2, overlaps)
    corrections[s2][s1] = get_correction(s2, s1, overlaps)


# ### Traverse the corrections graph recursively

def get_overlap(data1, data2):
    set1 = {*(tuple(c for c in r) for r in data1)}
    set2 = {*(tuple(c for c in r) for r in data2)}
    return set1 & set2


def correct_back(cur):
    print(f'\nEntering {cur}')
    corrected = {}
    to_descend = []
    for sid, correction in corrections[cur].items():
        if sid in visited:
            print(f'Already done {sid}')
            continue
        else:
            print(f'Correcting from {sid} to {cur} directly')
            corrected[sid] = correction(scans[sid])
            visited.add(sid)
            to_descend.append(sid)
    print(len(corrected))
            
    for sid in to_descend:
        print(f'Descending to find corrections to {sid}')
        to_correct = correct_back(sid)
        print(f'Back at {cur}')
        print(f'Got back {len(to_correct)} corrections')
        for target, data in to_correct.items():
            print(f'Correcting {target} inherited from {sid} to {cur}')
            corrected[target] = corrections[cur][sid](data)
    print(len(corrected))
    print()
            
    return corrected


visited = {0}
new_scans = correct_back(0)
new_scans[0] = scans[0]


full_map = set()
for data in new_scans.values():
    full_map |= {*(tuple(c for c in r) for r in data)}


len(full_map)


# # Part 2

def get_offset(s1, s2, overlaps, scans=scans):
    reverse = False
    if s1 > s2:
        s1, s2 = s2, s1
        reverse = True
        
    if (s1, s2) not in overlaps:
        print('Not an overlapping pair')
        return None
    
    matches, rot = overlaps[(s1, s2)]
    bid1, bid2, _ = matches[0]
    b1 = scans[s1][bid1]
    b2 = scans[s2][bid2]
    offset = b1 - rot.dot(b2)
    
    if reverse:
        rot = np.linalg.inv(rot)
        offset = b2 - rot.dot(b1)
    
    def correction(pos):
        corrected = rot.dot(pos) + offset
        return corrected
    
    return correction


offsets = defaultdict(dict)
for s1, s2 in overlaps.keys():
    offsets[s1][s2] = get_offset(s1, s2, overlaps)
    offsets[s2][s1] = get_offset(s2, s1, overlaps)


offsets[14]


def get_positions(cur):
    print(f'\nEntering {cur}')
    positions = {}
    to_descend = []
    for sid, correction in offsets[cur].items():
        if sid in visited:
            print(f'Already done {sid}')
            continue
        else:
            print(f'Correcting from {sid} to {cur} directly')
            positions[sid] = correction(np.array((0,0,0)))
            #print(positions[sid])
            visited.add(sid)
            to_descend.append(sid)
    print(len(positions))
            
    for sid in to_descend:
        print(f'Descending to find offsets to {sid}')
        to_correct = get_positions(sid)
        print(f'Back at {cur}')
        print(f'Got back {len(to_correct)} offsets')
        for target, pos in to_correct.items():
            print(f'Correcting {target} inherited from {sid} to {cur}')
            positions[target] = offsets[cur][sid](pos)
    print(len(positions))
    print()
            
    return positions


visited = {0}
positions = get_positions(0)
positions[0] = np.array((0, 0, 0))


max_dist = 0
pair = None
for i, pos1 in positions.items():
    for j, pos2 in positions.items():
        if i != j:
            dist = sum(abs(pos1 - pos2))
            if dist > max_dist:
                max_dist = dist
                pair = i, j


pair, max_dist




