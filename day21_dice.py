#!/usr/bin/env python
# coding: utf-8

from itertools import cycle
from collections import Counter


p1_start = 8
p2_start = 7


# # Part 1

die = cycle(range(100))
def roll():
    res = 0
    for i in range(3):
        res += (next(die) + 1)
    return res


def move(pos, steps):
    return (pos + steps -1) % 10 + 1


die = cycle(range(100))

p1 = p1_start
p2 = p2_start
points1, points2 = 0, 0
rolls = 0

for i in range(int(1e6)):
    rolls += 3
    steps = roll()
    
    if i%2 == 0:
        p1 = move(p1, steps)
        points1 += p1
        #print(f'p1 moves {steps} steps to {p1} for a total of {points1}')
        if points1 >= 1000:
            print('player 1 wins')
            break
    else:
        p2 = move(p2, steps)
        points2 += p2
        #print(f'p2 moves {steps} steps to {p2} for a total of {points2}')
        if points2 >= 1000:
            print('player 2 wins')
            break


print(points1, points2, rolls)


min(points2, points1)*rolls


# # Part 2

from collections import namedtuple


threshold = 21

# The state of one game is a tuple
# position of player one and two, score of one and two
State = namedtuple('State', ['pos1', 'pos2', 'score1', 'score2'])
# We will keep a Counter of all universes for a given state
states = Counter([State(p1_start, p2_start, 0, 0)])
wins = Counter()

while len(states) > 0:
    for player in range(2):
        new_states = Counter()
        # roll the dice 3 times
        for s1 in range(1, 4):
            for s2 in range(1, 4):
                for s3 in range(1, 4):
                    steps = s1 + s2 + s3
                    for state, count in states.items():
                        pos = state[player]
                        score = state[player+2]
                        new_pos = move(pos, steps)
                        
                        if score + new_pos >= threshold:
                            wins[player] += count
                        else:
                            new_state = list(state)
                            new_state[player] = new_pos
                            new_state[player+2] = score + new_pos
                            new_state = State(*new_state)
                            #print(new_state)
                            new_states[new_state] += count
        states = new_states
    print(len(states), 'different states')
    print(sum(states.values()), 'universes')


wins.most_common()


states


# ## Old, wrong approach
# We moved players independently

def step(pawns):
    """Do one step for one player"""
    new_pawns = {pos: Counter() for pos in range(1,11)}
    # For each outcome of the die
    for s1 in range(1, 4):
        for s2 in range(1, 4):
            for s3 in range(1, 4):
                steps = s1 + s2 + s3
                #print(f'rolled {s1}+{s2}+{s3}={steps}')
                # For each current position of the board
                for pos in pawns:
                    # Move the number of steps
                    new_pos = move(pos, steps)
                    #print(f'moving from {pos} to {new_pos}')
                    # Add the score 
                    new_pawns[new_pos].update(
                        {score+new_pos: count for
                         score, count in pawns[pos].items()}
                    )
                #print(f'current scores:', new_pawns)
                    
    return new_pawns


# player 1 is index 0, player 2 is 1
# for each player, for each positions, for each score
# we keep the number of pawns
player_pawns = {}
for player in range(2):
    player_pawns[player] = {}
    for pos in range(1, 11):
        player_pawns[player][pos] = Counter()
        
#player_pawns[0][p1_start][0] = 1
#player_pawns[1][p2_start][0] = 1
player_pawns[0][4][0] = 1
player_pawns[1][8][0] = 1

wins = Counter()
print(player_pawns)
print()

for i in range(10):
    for player in range(2):
        #print(f'>>> PLAYER {player+1}')
        pawns = step(player_pawns[player])
        player_pawns[player] = pawns
        #print()
        
    # TODO: Check if anyone won and stop incrementing for them
    for player, pawns in player_pawns.items():
        for pos, scores in pawns.items():
            for score in scores.copy():
                if score >= 21:
                     wins[player] += scores.pop(score)
    print(player_pawns)
    print()


wins


pos = 8
score = 0
for i in range(10):
    pos = move(pos, 3)
    print(pos)
    score += pos
    if score >= 21:
        break
score

