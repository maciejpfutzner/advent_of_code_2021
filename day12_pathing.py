#!/usr/bin/env python
# coding: utf-8

data_filename = 'day12_caves.txt'
with open(data_filename) as datafile:
    data = [l.strip() for l in datafile.readlines()]


data


class Cave:
    def __init__(self, name):
        self.name = name
        self.is_big = False
        self.connections = set()
        
        if name == name.upper():
            self.is_big = True
            
    def __repr__(self):
        return f"Cave('{self.name}')"


class CaveDict(dict):
    def __init__(self, *args):
        dict.__init__(self, args)
        
    def __missing__(self, key):
        self[key] = Cave(key)
        return self[key]


caves = CaveDict()

for edge in data:
    idx = edge.find('-')
    name1 = edge[:idx]
    cave1 = caves[name1]
    
    name2 = edge[idx+1:]
    cave2 = caves[name2]
    
    cave1.connections.add(cave2)
    cave2.connections.add(cave1)


start = caves['start']
end = caves['end']


# # Part 1

class Path():
    def __init__(self, path):
        self.path = tuple(c.name for c in path)
    
    def __repr__(self):
        return ' -> '.join(self.path)
    
    def __eq__(self, other):
        return self.path == other.path
    
    def __hash__(self):
        return hash(self.path)


path = Path(cave_system.values())
path





def go_back():
    try:
        n1 = len(visited)
        n2 = len(options_list)
        current = visited.pop()
        options = options_list.pop()
        if not (options < current.connections):
            print('options', options)
            print('connections', current.connections)
            raise ValueError('Options not a subset of connections')

        #print('current', current)
        #print('options', options)
        #print('visited', visited)
        return current, options
    
    except IndexError:
        print('visited len', n1)
        print('options list len', n2)
        raise


paths = set()
options_list = []
visited = []

current = start
options = current.connections.copy()
print('current', current)
print('options', options)

# Terminate when we're back at the start with no options
while current != start or options:
    #print()
    
    if current == end:
        print('>>> found new path to exit')
        path = Path(visited)
        print(path)
        if path in paths:
            raise KeyError('Path already present - how?')
        paths.add(path)
        
        print('go back and try again\n')
        current, options = go_back()
        continue
        
    if options:
        #print('next step')
        try:
            next_ = options.pop()
            while next_ in visited and not next_.is_big:
                #print(next_, 'already visited')
                next_ = options.pop()
        except KeyError:
            #print('no more options, go back')
            current, options = go_back()
            continue
        
        visited.append(current)
        options_list.append(options)
        
        # new options
        current = next_
        options = current.connections.copy()
        
        #print('current', current)
        #print('options', options)
        #print('visited', visited)
    else:
        #print('no options, go_back')
        current, options = go_back()


len(paths)


# # Part 2

small_caves = [c for c in caves if not caves[c].is_big]
small_caves.remove('start')
small_caves.remove('end')
small_caves.append(None)
small_caves


def generate_caves(caves, duplicated=None):
    new_caves = {n: Cave(n) for n in caves}
    for name, cave in new_caves.items():
        conn_names = [conn.name for conn in caves[name].connections]
        for conn_name in conn_names:
            cave.connections.add(new_caves[conn_name])
    
    if duplicated is not None:
        dup_name = duplicated + '1'
        dup_cave = Cave(duplicated)
        new_caves[dup_name] = dup_cave
        dup_cave.connections = new_caves[duplicated].connections
        for conn in dup_cave.connections:
            conn.connections.add(dup_cave)
        
    return new_caves


cave_systems = [generate_caves(caves, dupl) for dupl in small_caves]


def find_paths(start, end):
    def go_back():
        n1 = len(visited)
        n2 = len(options_list)
        current = visited.pop()
        options = options_list.pop()
        return current, options
        
    paths = set()
    options_list = []
    visited = []

    current = start
    options = current.connections.copy()
    #print('current', current)
    #print('options', options)

    # Terminate when we're back at the start with no options
    while current != start or options:
        if current == end:
            #print('>>> found new path to exit')
            path = Path(visited)
            #print(path)
            paths.add(path)

            #print('go back and try again\n')
            current, options = go_back()
            continue

        if options:
            #print('next step')
            try:
                next_ = options.pop()
                while next_ in visited and not next_.is_big:
                    #print(next_, 'already visited')
                    next_ = options.pop()
            except KeyError:
                #print('no more options, go back')
                current, options = go_back()
                continue

            visited.append(current)
            options_list.append(options)

            # new options
            current = next_
            options = current.connections.copy()

            #print('current', current)
            #print('options', options)
            #print('visited', visited)
        else:
            #print('no options, go_back')
            current, options = go_back()
            
    return paths


all_paths = set()
n_paths = 0

for cave_system in cave_systems:
    print('new cave system')
    print([c.name for c in cave_system.values()])
    start = cave_system['start']
    end = cave_system['end']
    paths = find_paths(start, end)
    print(len(paths))
    n_paths += len(paths)
    all_paths |= paths


n_paths


len(all_paths)







