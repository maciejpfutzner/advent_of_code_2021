#!/usr/bin/env python
# coding: utf-8

import re
from collections import defaultdict


filename = 'day2_dive.txt'


with open(filename, 'r') as infile:
    commands = infile.readlines()


# # Part 1

parsed = [re.match(r'(\w+) (\d+).*', cmd).groups() for cmd in commands]


totals = defaultdict(int)


for cmd, val in parsed:
    totals[cmd] += int(val)


totals


fwd = totals['forward']
depth = totals['down'] - totals['up']
print(fwd)
print(depth)


fwd*depth


# # Part 2

class State:
    def __init__(self):
        self.fwd = 0
        self.depth = 0
        self.aim = 0
        
    def run_cmd(self, command, val):
        if command == 'up':
            #self.depth -= val
            self.aim -= val
        elif command == 'down':
            #self.depth += val
            self.aim += val
        elif command == 'forward':
            self.fwd += val
            self.depth += val*self.aim
            
    def __str__(self):
        return f'fwd = {self.fwd}, depth = {self.depth}, aim = {self.aim}'


s = State()

for cmd, val in parsed:
    s.run_cmd(cmd, int(val))
    print(cmd, val)
    print(s)


s.fwd * s.depth




