#!/usr/bin/env python
# coding: utf-8

from ast import literal_eval
from copy import deepcopy
from collections import deque
from math import ceil, floor


with open('day18_snailfish.txt') as dfile:
    numbers = [l.strip() for l in dfile]
numbers = [literal_eval(l) for l in numbers]


# # Reimplement with a class
class Snumber():
    def __init__(self, iterable):
        self.number = deepcopy(iterable)
        self.indices = self.get_indices(self.number)
        
    def __eq__(self, other):
        return self.number == other.number
    
    def __repr__(self):
        return f'Snumber({repr(self.number)})'
    
    def __getitem__(self, ind):
        try:
            n_list = self.number
            for i in ind:
                n_list = n_list[i]
            return n_list
        except IndexError:
            return None
        
    def __setitem__(self, ind, value):
        n_list = self.number
        for i in ind[:-1]:
            n_list = n_list[i]
        n_list[ind[-1]] = value
        # Redo indices, they might have changed
        self.indices = self.get_indices(self.number)
        
    def find_left(self, location):
        """Find the int to the left of location

        Args:
            location - list of indices for the desired digit
        Returns:
            list of indices for the left-hand digit or None
        """
        where = self.indices.index(location)
        if where > 0:
            return self.indices[where-1]
        else:
            return None
        
    def find_right(self, location):
        """Find the number to the right of location

        Args:
            location - list of indices for the desired digit
        Returns:
            list of indices for the left-hand digit or None
        """
        where = self.indices.index(location)
        if where < len(self.indices) - 1:
            return self.indices[where+1]
        else:
            return None
        
    def explode(self):
        # Find leftmost pair 4 levels deep
        location = None
        for inds in self.indices:
            if len(inds) > 4:
                location = inds[:-1]
                break
        if location is None:
            return False
        else:
            n1, n2 = self[location]
            
            # Increment left neighbour (if found)
            left = self.find_left(location+[0])
            if left:
                self[left] += n1

            # Increment right neighbour (if found)
            right = self.find_right(location+[1])
            if right:
                self[right] += n2

            # Replace pair with 0
            self[location] = 0
            return True
        
    def split(self):
        for location in self.indices:
            digit = self[location]
            if digit > 9:
                n1 = floor(digit/2)
                n2 = ceil(digit/2)
                self[location] = [n1, n2]
                return True
        return False
    
    def reduce(self):
        changed = self.explode()
        if changed:
            return self.reduce()
        else:
            changed = self.split()
            if changed:
                return self.reduce()
            else:
                return self
            
    def __add__(self, other):
        new = Snumber([self.number, other.number])
        new.reduce()
        return new
    
    def magnitude(self):
        def mag(n):
            if type(n) is int:
                return n
            else:
                n1, n2 = n
                return 3*mag(n1) + 2*mag(n2)
        return mag(self.number)
    
    @staticmethod
    def get_indices(n):
        if type(n) is int:
            return [[]]

        else:
            indices = []
            for i in range(2):
                for ids in Snumber.get_indices(n[i]):
                    indices.append([i] + ids)

            return indices


# ### Part 1

snumbers = [Snumber(n) for n in numbers]

total = snumbers[0]
for n in snumbers[1:]:
    total = total + n
    
print(total.magnitude())


# ### Part 2

max_mag = 0

N = len(snumbers)
for i in range(N):
    for j in range(i+1, N):
        n1 = snumbers[i]
        n2 = snumbers[j]
        mag = (n1 + n2).magnitude()
        if mag > max_mag:
            max_mag = mag
            
        mag = (n2 + n1).magnitude()
        if mag > max_mag:
            max_mag = mag

print(max_mag)




