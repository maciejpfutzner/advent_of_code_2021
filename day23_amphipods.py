#!/usr/bin/env python
# coding: utf-8

from heapq import heappop, heappush
from copy import deepcopy


start_map_txt = """#############
#...........#
###D#A#C#D###
  #B#C#B#A#
  #########"""

end_map_txt = """#############
#...........#
###A#B#C#D###
  #A#B#C#D#
  #########"""


# # Rules
# - Energy
#     - Amber amphipods require 1 energy per step
#     - Bronze amphipods require 10 energy
#     - Copper amphipods require 100
#     - Desert ones require 1000
# - Amphipods will never stop on the space immediately outside any room. They can move into that space so long as they immediately continue moving.
#     - (Specifically, this refers to the four open spaces in the hallway that are directly above an amphipod starting position.)
# - Amphipods will never move from the hallway into a room unless that room is their destination room and that room contains no amphipods which do not also have that room as their own destination. If an amphipod's starting room is not its destination room, it can stay in that room until it leaves the room.
#     - (For example, an Amber amphipod will not move from the hallway into the right three rooms, and will only move into the leftmost room if that room is empty or if it only contains other Amber amphipods.)
# - Once an amphipod stops moving in the hallway, it will stay in that spot until it can move into a room.
#     - (That is, once any amphipod starts moving, any other amphipods currently in the hallway are locked in place and will not move again until they can move fully into a room.)

# # Part 1

cost_dict = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
room_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4}


start_rooms = [['B', 'D'], ['C', 'A'], ['B', 'C'], ['A', 'D']]
# A stop at index i is to the left of room i-1 (room_id i)
start_stops = ['.']*7


end_rooms = [['A', 'A'], ['B', 'B'], ['C', 'C'], ['D', 'D']]


# Legal moves are:
# - Getting out of a room and into a stop
#     - Without passing any other amphipod
# - Getting into a room from a stop
#     - Without passing anyone
#     - Only to destination room (empty or with 1 correct guy)

def find_room_to_stop(room_id, stops):
    """Find possible moves into the corridor
    
    Find all possible routes out of a given room into the corridor,
    without passing others. Return a list of the possible corridor indices
    and the required number of steps.
    """
    res_sids, res_steps = [], []
    # Check where to go to the left
    for sid in range(room_id, -1, -1):
        if stops[sid] != '.':
            # Cannot go anymore
            break
        else:
            # This is an empty space, return it as possible
            res_sids.append(sid)
            # It takes 2 steps per stop, except the leftmost one
            n_steps = 2* (room_id - sid + 1)
            if sid == 0:
                n_steps -= 1
            res_steps.append(n_steps)
            
    # Check where to go to the right
    for sid in range(room_id+1, len(stops), 1):
        if stops[sid] != '.':
            # Cannot go anymore
            break
        else:
            # This is an empty space, return it as possible
            res_sids.append(sid)
            # It takes 2 steps per stop, except the leftmost one
            n_steps = 2* (sid - room_id)
            if sid == len(stops) - 1:
                n_steps -= 1
            res_steps.append(n_steps)
    
    return zip(res_sids, res_steps)


def check_stop_to_room(stop_id, room_id, stops):
    # Are there any guys between the stop and the room entrance?
    if stop_id <= room_id:
        # Room is on the right of the stop
        for sid in range(stop_id+1, room_id+1):
            if stops[sid] != '.':
                # If any of the stops is occupied, you can't go
                return False
        n_steps = (room_id - stop_id + 1) * 2
        if stop_id == 0:
            n_steps -= 1
        return n_steps
    
    else:
        # Room is on the left of the stop
        for sid in range(room_id+1, stop_id):
            if stops[sid] != '.':
                # If any of the stops is occupied, you can't go
                return False
        n_steps = (stop_id - room_id) * 2
        if stop_id == len(stops) - 1:
            n_steps -= 1
        return n_steps


def plot(state):
    rooms, stops = state
    stops = stops[0] + '.'.join(stops[1:-1]) + stops[-1]
    print('#############')
    print(f'#{"".join(stops)}#')
    print(f'###{rooms[0][1]}#{rooms[1][1]}#{rooms[2][1]}#{rooms[3][1]}###')
    print(f'  #{rooms[0][0]}#{rooms[1][0]}#{rooms[2][0]}#{rooms[3][0]}###')
    print('  #########')


def get_legal_moves(rooms, stops, start_cost, queue):
    # Find possible moves out of a room
    for i, room in enumerate(rooms):
        rid = i+1
        if (guy := room[1]) != '.':
            extra_step = 0
        elif (guy := room[0]) != '.':
            extra_step = 1
        else:
            # Can't move out of this room
            continue
            
        # the top or the bottom guy can go into a stop (that's not blocked)
        moves = find_room_to_stop(rid, stops)
        for new_stop_pos, n_steps in moves:
            # New rooms will have an empty spot
            new_rooms = deepcopy(rooms)
            if extra_step:
                new_rooms[rid-1] = ['.', '.']
            else:
                new_rooms[rid-1] = [room[0], '.']
            # New corridor will have the guy ins new spot
            new_stops = stops.copy()
            new_stops[new_stop_pos] = guy
            # Cost is number of steps (+1 one if from back) times energy
            cost = start_cost + (n_steps + extra_step) * cost_dict[guy]
            heappush(queue, (cost, new_rooms, new_stops))
            
    # Find possible moves into a room
    for sid, guy in enumerate(stops):
        if guy != '.':
            # There's a guy in the corridor, see if his room is ready
            rid = room_dict[guy]
            dest_room = rooms[rid - 1]
            if dest_room == ['.', '.']:
                extra_step = 1
            elif dest_room == [guy, '.']:
                extra_step = 0
            else:
                # This guy cannot move into any room
                break
                
            # Is there a clear way and how many steps
            n_steps = check_stop_to_room(sid, rid, stops)
            if n_steps:
                new_rooms = deepcopy(rooms)
                if extra_step:
                    new_rooms[rid-1] = [guy, '.']
                else:
                    new_rooms[rid-1] = [guy, guy]
                # New corridor will have an empty spot
                new_stops = stops.copy()
                new_stops[sid] = '.'
                # Cost is number of steps (+1 one if from back) times energy
                cost = start_cost + (n_steps + extra_step) * cost_dict[guy]
                heappush(queue, (cost, new_rooms, new_stops))


end_state = (tuple(tuple(r) for r in end_rooms), tuple(start_stops))
#end_state in visited


queue = []
visited = set()
heappush(queue, (0, start_rooms, start_stops))

while queue:
    cost, rooms, stops = heappop(queue)
    state = (tuple(tuple(r) for r in rooms), tuple(stops))
    if state == end_state:
        print(f"We're finished, the cost is {cost}")
        break
    if state not in visited:
        visited.add(state)
        get_legal_moves(rooms, stops, cost, queue)


plot(state)


# # Part 2

start_map_txt = """#############
#...........#
###D#A#C#D###
  #D#C#B#A#
  #D#B#A#C#
  #B#C#B#A#
  #########"""

end_map_txt = """#############
#...........#
###A#B#C#D###
  #A#B#C#D#
  #A#B#C#D#
  #A#B#C#D#
  #########"""


start_rooms = [['D', 'D', 'D', 'B'], ['A', 'C', 'B', 'C'],
               ['C', 'B', 'A', 'B'], ['D', 'A', 'C', 'A']]
# A stop at index i is to the left of room i-1 (room_id i)
start_stops = ['.']*7


end_rooms = [['A', 'A', 'A', 'A'], ['B', 'B', 'B', 'B'],
             ['C', 'C', 'C', 'C'], ['D', 'D', 'D', 'D']]
end_state = (tuple(tuple(r) for r in end_rooms), tuple(start_stops))


def plot2(state):
    rooms, stops = state
    stops = stops[0] + '.'.join(stops[1:-1]) + stops[-1]
    print('#############')
    print(f'#{"".join(stops)}#')
    print(f'###{rooms[0][0]}#{rooms[1][0]}#{rooms[2][0]}#{rooms[3][0]}###')
    print(f'  #{rooms[0][1]}#{rooms[1][1]}#{rooms[2][1]}#{rooms[3][1]}#')
    print(f'  #{rooms[0][2]}#{rooms[1][2]}#{rooms[2][2]}#{rooms[3][2]}#')
    print(f'  #{rooms[0][3]}#{rooms[1][3]}#{rooms[2][3]}#{rooms[3][3]}#')
    print('  #########')


def get_legal_moves2(rooms, stops):
    next_moves = []
    
    # Find possible moves out of a room
    for rid, room in enumerate(rooms):
        # Find the top-most guy in the room (and index)
        for i, guy in enumerate(room):
            if guy != '.':
                break
                
        # If room was empty, move to the next one
        if guy == '.':
            continue
            
        # the top-most guy can go into a stop (that's not blocked)
        moves = find_room_to_stop(rid+1, stops)
        
        for new_stop_pos, n_steps in moves:
            # New rooms will have a new empty spot
            new_rooms = deepcopy(rooms)
            new_rooms[rid][i] = '.'
            # New corridor will have the guy ins new spot
            new_stops = stops.copy()
            new_stops[new_stop_pos] = guy
            # Cost is number of steps (+ depth of occupied spot) times energy
            cost = (n_steps + i) * cost_dict[guy]
            next_moves.append((cost, new_rooms, new_stops))
            #print('Getting out of the room')
            #plot2((rooms, stops))
            #print('Next state')
            #plot2((new_rooms, new_stops))
            #print()
            
    # Find possible moves into a room
    for sid, guy in enumerate(stops):
        if guy != '.':
            # There's a guy in the corridor, see if his room is ready
            rid = room_dict[guy] - 1
            dest_room = rooms[rid]
            
            if set(dest_room) <= {'.', guy}:
                #print(f'Room {rid+1} was supposed to be ready for {guy}')
                #print(dest_room)
                # Good room, find index of top-most empty spot
                i = len(dest_room) - dest_room[::-1].index('.') -1
            else:
                # This guy cannot move into a room
                continue
                
            # Is there a clear way and how many steps
            n_steps = check_stop_to_room(sid, rid+1, stops)
            if n_steps:
                new_rooms = deepcopy(rooms)
                new_rooms[rid][i] = guy
                # New corridor will have an empty spot
                new_stops = stops.copy()
                new_stops[sid] = '.'
                # Cost is number of steps (+1 one if from back) times energy
                cost = (n_steps + i) * cost_dict[guy]
                next_moves.append((cost, new_rooms, new_stops))
                #print('Getting into the room')
                #plot2((rooms, stops))
                #print('Next state')
                #plot2((new_rooms, new_stops))
                #print()
    return next_moves


queue = []
visited = set()
heappush(queue, (0, start_rooms, start_stops))

while queue:
    cost, rooms, stops = heappop(queue)
    state = (tuple(tuple(r) for r in rooms), tuple(stops))
    #plot2(state)
    #print()
    if state == end_state:
        print(f"We're finished, the cost is {cost}")
        break
    elif state not in visited:
        visited.add(state)
        moves = get_legal_moves2(rooms, stops)
        for new_cost, new_rooms, new_stops in moves:
            heappush(queue, (cost+new_cost, new_rooms, new_stops))
        #if len(moves) == 0:
        #    print('No more legal moves')
        #    plot2(state)
        #    print()




