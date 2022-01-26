#!/usr/bin/env python
# coding: utf-8

import numpy as np


filename = 'day3_binary.txt'


with open(filename, 'r') as infile:
    bits = infile.readlines()


# # Part 1 

bits = np.array([[int(b) for b in bit.strip()] for bit in bits ])


bits.shape


gamma_arr = np.median(bits, axis=0).astype(int)


gamma = int(''.join([str(b) for b in gamma_arr]), 2)
gamma


epsilon_arr = (~gamma_arr.astype(bool)).astype(int)


epsilon = int(''.join([str(b) for b in epsilon_arr]), 2)
epsilon


gamma*epsilon


# # Part 2

def get_rating(bits, invert=False):
    bits2 = bits
    for iteration in range(bits.shape[1]):
        most_common = np.median(bits2[:,iteration])
        if most_common == .5:
            most_common = 1
        if invert:
            most_common = 1 - most_common
        sel = bits2[:,iteration] == most_common
        bits2 = bits2[sel]
        if bits2.shape[0] == 1:
            break
    val = int(''.join([str(b) for b in bits2[0]]), 2)
    return val


ox_rating = get_rating(bits)
ox_rating


co2_rating = get_rating(bits, invert=True)
co2_rating


ox_rating * co2_rating




