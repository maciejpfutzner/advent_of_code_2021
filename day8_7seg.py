#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np


data_fname = 'day8_7seg.txt'
with open(data_fname, 'r') as datafile:
    data = [l.strip() for l in datafile.readlines()]


pattern = '[abcdefg]+'


inputs = []
outputs = []

for d in data:
    inputs_txt = d[:d.find('|')]
    inputs.append(re.findall(pattern, inputs_txt))
                  
    outputs_txt = d[d.find('|'):]
    outputs.append(re.findall(pattern, outputs_txt))


# # Part 1

lengths = [len(d) for digits in outputs for d in digits]


np.isin(lengths, [2, 3, 4, 7]).sum()


# # Part 2 - the hard part

def map_easy_digits(digits):
    len_digits = {len(d): d for d in digits}
    len_mapping = {2: 1, 3: 7, 4: 4, 7: 8}
    result = {}
    for length,digit in len_digits.items():
        if length in len_mapping:
            result[digit] = len_mapping[length]
    return result


def find_mapping(inputs):
    mapping = map_easy_digits(inputs)
    rev_mapping = {v:set(k) for k,v in mapping.items()}

    lengths = {d: len(d) for d in inputs}
    # 2, 3 and 5
    seg5 = [d for d in lengths if lengths[d] == 5]

    # 6, 9, 0
    seg6 = [d for d in lengths if lengths[d] == 6]

    # Find 9
    for d in seg6:
        if rev_mapping[4] < set(d):
            mapping[d] = 9
            rev_mapping[9] = set(d)
            seg6.remove(d)
            break

    # Find 3
    for d in seg5:
        if rev_mapping[1] < set(d):
            mapping[d] = 3
            rev_mapping[3] = set(d)
            seg5.remove(d)
            break

    # Find 0
    for d in seg6:
        if rev_mapping[1] < set(d): # and 4 not in
            mapping[d] = 0
            rev_mapping[0] = set(d)
            seg6.remove(d)
            break

    # The remaining seg6 digit is 6
    mapping[seg6[0]] = 6

    # Find 5
    for d in seg5:
        if set(d) < rev_mapping[9]:
            mapping[d] = 5
            rev_mapping[5] = set(d)
            seg5.remove(d)
            break

    # The remaining seg5 digit is 2
    mapping[seg5[0]] = 2
    
    return mapping


ex = 'acedgfb cdfbe gcdfa fbcad dab cefabd cdfgeb eafb cagedb ab'.split(' ')
find_mapping(ex)


def map_digits(segments, mapping):
    for pattern, value in mapping.items():
        if set(segments) == set(pattern):
            return value


decoded_outputs = []
for input_, output in zip(inputs, outputs):
    mapping = find_mapping(input_)
    output_str = ''.join([str(map_digits(o, mapping)) for o in output])
    decoded_outputs.append(int(output_str))


sum(decoded_outputs)


outputs[0]




