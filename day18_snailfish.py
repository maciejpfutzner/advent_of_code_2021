#!/usr/bin/env python
# coding: utf-8

from ast import literal_eval
from copy import deepcopy
from collections import deque
from math import ceil, floor


with open('day18_snailfish.txt') as dfile:
    numbers = [l.strip() for l in dfile]
numbers = [literal_eval(l) for l in numbers]


# # Part 1

def access(nested_list, indices):
    try:
        for i in indices:
            nested_list = nested_list[i]
        return nested_list
    except IndexError:
        return None


def get_indices(n):
    if type(n) is int:
        return [[]]
    
    else:
        indices = []
        for i in range(2):
            for ids in get_indices(n[i]):
                indices.append([i] + ids)
            
        return indices


n = [[2, [[7, 4], [5, [3, 9]]]], [[[[1, 4], [0, 1]], 4], [3, [8, 5]]]]

print(n)
for ii in get_indices(n):
    print(f'{access(n, ii)}: {ii}')


def find_left(n, location):
    """Find the number to the left of location
    
    Args:
        n - snailfish number
        location - list of indices for the desired digit
    Returns:
        list of indices for the left-hand digit or None
    """
    all_ids = get_indices(n)
    where = all_ids.index(location)
    
    if where > 0:
        return all_ids[where-1]
    else:
        return None


n = [[2, [[7, 4], [5, [3, 9]]]], [[[[1, 4], [0, 1]], 4], [3, [8, 5]]]]
all_ids = get_indices(n)

loc = all_ids[1]
print(access(n, loc))
ii = find_left(n, loc)
print(ii)
print(access(n, ii))


def find_right(n, location):
    """Find the number to the right of location
    
    Args:
        n - snailfish number
        location - list of indices for the desired digit
    Returns:
        list of indices for the left-hand digit or None
    """
    all_ids = get_indices(n)
    where = all_ids.index(location)
    
    if where < len(all_ids) - 1:
        return all_ids[where+1]
    else:
        return None


def explode_next(n):
    # Find leftmost pair 4 levels deep
    location = None
    for inds in get_indices(n):
        if len(inds) > 4:
            location = inds[:-1]
            break
        
    if location is None:
        return n, False
    else:
        #print(f'exploding {access(n, location)} at {location}')
        n1, n2 = access(n, location)
        
        # Increment left neighbour (if found)
        left = find_left(n, location+[0])
        if left:
            #print(f'found left digit {access(n, left)}, incrementing by {n1}')
            i1, i2 = left[:-1], left[-1]
            access(n, i1)[i2] += n1
            
        # Increment right neighbour (if found)
        right = find_right(n, location+[1])
        if right:
            #print(f'found right digit {access(n, right)}, incrementing by {n2}')
            i1, i2 = right[:-1], right[-1]
            access(n, i1)[i2] += n2
        
        # Replace pair with 0
        access(n, location[:-1])[location[-1]] = 0
    
        return n, True


n = [[2, [[7, 4], [5, [3, 9]]]], [[[[1, 4], [0, 1]], 4], [3, [8, 5]]]]
print(n)
explode_next(n)


n = [[[[[9,8],1],2],3],4]
n,_ = explode_next(n)
n == [[[[0,9],2],3],4]


n = [[3,[2,[1,[7,3]]]],[6,[5,[4,[3,2]]]]]
n,_ = explode_next(n)
n == [[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]


def split_next(n):
    for location in get_indices(n):
        digit = access(n, location)
        if digit > 9:
            n1 = floor(digit/2)
            n2 = ceil(digit/2)
            #print(f'splitting {digit} into {[n1, n2]}')
            access(n, location[:-1])[location[-1]] = [n1, n2]
            return n, True
    return n, False


n = [[[[0,7],4],[15,[0,13]]],[1,1]]
n, _ = split_next(n)
print(n == [[[[0,7],4],[[7,8],[0,13]]],[1,1]])

n, _ = split_next(n)
n == [[[[0,7],4],[[7,8],[0,[6,7]]]],[1,1]]


def reduce(n):
    # 1. Assume explosions necessary
    # 2. Find next explosion
    # 3. If none, go to splits (5)
    # 4. If yes, apply and return to 2.
    # 5. Find next split
    # 6. If none, return number
    # 7. If yes, apply and return to 2.
    while True:
        #print()
        #print(n)
        n, changed = explode_next(n)
        if not changed:
            to_split = True
            while True:
                n, changed = split_next(n)
                if changed:
                    break
                else:
                    return n


n = [[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]]
n = reduce(n)
print(n)
n == [[[[0,7],4],[[7,8],[6,0]]],[8,1]]


def add(n1, n2):
    n = [n1, n2]
    return reduce(n)


n1 = [[[[4,3],4],4],[7,[[8,4],9]]]
n2 = [1,1]
n = add(n1, n2)
n == [[[[0,7],4],[[7,8],[6,0]]],[8,1]]





example = """[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]
[[[5,[2,8]],4],[5,[[9,9],0]]]
[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]
[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]
[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]
[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]
[[[[5,4],[7,7]],8],[[8,3],8]]
[[9,3],[[9,9],[6,[4,9]]]]
[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]
[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]"""

nns = [literal_eval(l) for l in example.split()]


total = nns[0]
for n in nns[1:]:
    total = add(total, n)


total == [[[[6,6],[7,6]],[[7,7],[7,0]]],[[[7,7],[7,7]],[[7,8],[9,9]]]]


def magnitude(n, mult=3):
    if type(n) is int:
        return n
    else:
        n1, n2 = n
        return 3*magnitude(n1) + 2*magnitude(n2)


magnitude(total)


# ### Ok, let's calculate the actual total

with open('day18_snailfish.txt') as dfile:
    numbers = [l.strip() for l in dfile]
numbers = [literal_eval(l) for l in numbers]


total = numbers[0]
for n in numbers[1:]:
    total = add(total, n)


total


magnitude(total)





# # Part 2

with open('day18_snailfish.txt') as dfile:
    numbers = [l.strip() for l in dfile]
numbers = [literal_eval(l) for l in numbers]


len(numbers)


max_mag = 0

N = len(numbers)
for i in range(N):
    for j in range(i+1, N):
        n1 = deepcopy(numbers[i])
        n2 = deepcopy(numbers[j])
        mag = magnitude(add(n1, n2))
        if mag > max_mag:
            max_mag = mag
            
        n1 = deepcopy(numbers[i])
        n2 = deepcopy(numbers[j])
        mag = magnitude(add(n2, n1))
        if mag > max_mag:
            max_mag = mag


max_mag





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


n1 = Snumber([[[[4,3],4],4],[7,[[8,4],9]]])
n2 = Snumber([1,1])
n = n1 + n2
print(n == Snumber([[[[0,7],4],[[7,8],[6,0]]],[8,1]]))


n.magnitude()


# ## Tests

n = Snumber([[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]])
n.reduce()
print(n)
n == Snumber([[[[0,7],4],[[7,8],[6,0]]],[8,1]])


example = """[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]
[[[5,[2,8]],4],[5,[[9,9],0]]]
[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]
[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]
[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]
[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]
[[[[5,4],[7,7]],8],[[8,3],8]]
[[9,3],[[9,9],[6,[4,9]]]]
[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]
[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]"""

nns = [Snumber(literal_eval(l)) for l in example.split()]


total = nns[0]
for n in nns[1:]:
    total = total + n

print(total == Snumber([[[[6,6],[7,6]],[[7,7],[7,0]]],[[[7,7],[7,7]],[[7,8],[9,9]]]]))
print(total.magnitude())


# ### Part 1

snumbers = [Snumber(n) for n in numbers]

total = snumbers[0]
for n in snumbers[1:]:
    total = total + n
    
total.magnitude()


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

max_mag




