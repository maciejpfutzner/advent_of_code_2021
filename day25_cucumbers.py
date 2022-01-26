#!/usr/bin/env python
# coding: utf-8

import numpy as np


with open('day25_cucumbers.txt') as infile:
    start_positions = [[s for s in row.strip()] for row in infile]
start_positions = np.array(start_positions)


# # Part 1

def sample(board, n=10):
    print(board[:n, :n])


def move_east(board):
    x, y = np.where(board=='>')
    y_new = y+1
    y_new[y_new==board.shape[1]] = 0
    free_neigbour = (board[(x, y_new)] == '.')
    east_free = (x[free_neigbour], y[free_neigbour])
    east_next = (x[free_neigbour], y_new[free_neigbour])
    n_changes = len(east_free[0])
    
    board[east_free] = '.'
    board[east_next] = '>'
    return board, n_changes


def move_south(board):
    x, y = np.where(board=='v')
    x_new = x+1
    x_new[x_new==board.shape[0]] = 0
    free_neigbour = (board[(x_new, y)] == '.')
    south_free = (x[free_neigbour], y[free_neigbour])
    south_next = (x_new[free_neigbour], y[free_neigbour])
    n_changes = len(south_free[0])
    
    board[south_free] = '.'
    board[south_next] = 'v'
    return board, n_changes


def step(board):
    board, n_changes_e = move_east(board)
    board, n_changes_s = move_south(board)
    if n_changes_e + n_changes_s > 0:
        return board, False
    else:
        return board, True


board = start_positions.copy()
sample(board, 15)

finished = False
n_steps = 0
while not finished:
    n_steps += 1
    board, finished = step(board)

print()
sample(board, 15)
print()
print(n_steps, 'steps')




