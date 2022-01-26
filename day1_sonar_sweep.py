#!/usr/bin/env python
# coding: utf-8

import numpy as np


# # Part 1

data_filename = 'data_01.txt'


with open(data_filename, 'r') as data_file:
    data = [int(l) for l in data_file.readlines()]


data = np.array(data)


data[:10]


((data[1:] - data[:-1]) > 0).sum()


# # Part 2

data_new = np.stack([data[:-2], data[1:-1], data[2:]]).sum(axis=0)


((data_new[1:] - data_new[:-1]) > 0).sum()




