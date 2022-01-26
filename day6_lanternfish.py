#!/usr/bin/env python
# coding: utf-8

data_fname = 'day6_fish_initial.txt'
with open(data_fname, 'r') as datafile:
    initial = [int(l) for l in datafile.read().split(',')]


len(initial)


# # Part 1

class Fish:
    def __init__(self, age=8):
        self.age = age
    
    def spawn(self):
        self.age = 6
        return Fish()
    
    def tick(self):
        self.age -= 1
        if self.age < 0:
            return self.spawn()
        else:
            return
        
    def __repr__(self):
        return f'Fish({self.age})'


fishes = [Fish(age) for age in initial]

for i in range(0, 80):
    print(f'Starting day {i} with {len(fishes)} fishes')
    new_fishes = [f for f in [f.tick() for f in fishes] if f is not None]
    fishes.extend(new_fishes)
    print(f'Adding {len(new_fishes)} new fishes')


len(fishes)


# # Part 2
# Now we need to vectorise it

from collections import Counter, defaultdict


def tick(age_counter):
    # Remove 1 day from all ages
    new_counter = defaultdict(int, {a-1: n for a,n in age_counter.items()})
    
    # Remove all the overdue fishes
    spawning_fishes = new_counter.pop(-1, 0)
    
    # Add them back with counter 6
    new_counter[6] += spawning_fishes
    # And add the same number of new fishes
    new_counter[8] += spawning_fishes
    
    return new_counter


age_counter = Counter(initial)

for i in range(256):
    age_counter = tick(age_counter)


sum(age_counter.values())




