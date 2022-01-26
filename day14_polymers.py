#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import Counter, defaultdict


start = 'CBNBOKHVBONCPPBBCKVH'
data_filename = 'day14_polymers.txt'
with open(data_filename) as datafile:
    rules_txt = [l.strip() for l in datafile.readlines()]


rules = {}
for row in rules_txt:
    pair, insert = row.split(' -> ')
    rules[pair] = insert


rules


# # Part 1

def get_insertions(polymer):
    insertions = []
    for i in range(len(polymer) -1):
        insertions.append(rules[polymer[i:i+2]])
    return insertions


polymer = start

for i in range(10):
    insertions = get_insertions(polymer)
    new_poly = np.insert(list(polymer), range(1, len(polymer)), list(insertions))
    polymer = ''.join(new_poly)


polymer


counts = Counter(polymer)
most_common = counts.most_common()
most_common


most_common[0][1] - most_common[-1][1]


# # Part 2

#slen = len(start)
#for i in range(40):
#    slen += slen-1
#slen


letters = np.unique([pair[0] for pair in rules])
first = polymer[0]
last = polymer[-1]


pairs = {pair: 0 for pair in rules.keys()}
for i in range(len(start)-1):
    pairs[start[i:i+2]] += 1
    
for i in range(40):
    new_pairs = defaultdict(int)
    for pair, count in pairs.items():
        insert = rules[pair]
        new_pairs[pair[0] + insert] += count
        new_pairs[insert + pair[1]] += count
    pairs = new_pairs


counts = {}
for letter in letters:
    startwith = [(pair, pairs[pair]) for pair in pairs
                  if pair.startswith(letter) and not pair.endswith(letter)]
    n_starts = sum([ss[1] for ss in startwith])
    n_doubles = pairs[letter+letter]
    counts[letter] = n_starts + n_doubles
    if letter == last:
        counts[letter] += 1


most_common = Counter(counts).most_common()
most_common


most_common[0][1] - most_common[-1][1]




