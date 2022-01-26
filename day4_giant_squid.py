#!/usr/bin/env python
# coding: utf-8

import numpy as np


fname_numbers = 'day4_numbers.txt'
with open(fname_numbers, 'r') as infile:
    numbers = infile.read()
    numbers = [int(n) for n in numbers.split(',')]


numbers[:5]


fname_boards = 'day4_bingo_boards.txt'
with open(fname_boards, 'r') as infile:
    boards_txt = infile.readlines()
    #numbers = [int(n) for n in numbers.split(',')]


# # Part 1

class Board:
    def __init__(self, text=None):
        if text is not None:
            self.load_board(text)
    
    def load_board(self, text):
        self.board = np.array(
            [[int(n) for n in
              row.strip().replace('  ', ' ').split(' ')]
             for row in text])
        self.marked = set()
        
    def mark(self, number):
        if number in self.board:
            self.marked.add(number)
        return self
            
    def is_winner(self):
        #np.isin doesn't work with sets!!!
        test = np.isin(self.board, list(self.marked))
        if (np.any(test.mean(axis=0) == 1) or
            np.any(test.mean(axis=1) == 1)):
            return True
        else:
            return False
        
    def score(self, number):
        sum_ = self.board[~np.isin(self.board, list(self.marked))].sum()
        return sum_ * number
    
    def __repr__(self):
        BOLD = '\033[1m'
        END = '\033[0m'
        output = ''
        for row in self.board:
            for number in row:
                if number in self.marked:
                    output += f'{BOLD}{number: }{END} '
                else:
                    output += f'{number: } '
            output += '\n'
        return output


# ### Test the class

board_t = boards_txt[:5]
board_t


b = Board(board_t)
b.mark(84)
b.mark(85)
b.mark(94)
print(b.is_winner())
b.mark(24)
b.mark(44)
b.mark(52)
print(b.is_winner())


def mark_all_boards_check_winner(boards, number):
    for i, b in enumerate(boards):
        b.mark(number)
        if b.is_winner():
            score = b.score(number)
            print(f'Board {i} is the winner with the score {score}')
            print(b)
            return True
    return False


# ### Read all boards

boards = []
for i in range(0, len(boards_txt), 6):
    board_t = boards_txt[i:i+5]
    b = Board(board_t)
    boards.append(b)

for n in numbers:
    check = mark_all_boards_check_winner(boards, n)
    if check:
        break


# # Part 2

def mark_all_boards(boards, number):
    new_boards = []
    for i, b in enumerate(boards):
        b.mark(number)
        if b.is_winner():
            score = b.score(number)
            if score >0:
                print(f'Board {i} is the winner with the score {score}')
                print(b)
        else:
            new_boards.append(b)
    return new_boards


boards = []
for i in range(0, len(boards_txt), 6):
    board_t = boards_txt[i:i+5]
    b = Board(board_t)
    boards.append(b)

for n in numbers:
    print(n)
    boards = mark_all_boards(boards, n)
    if not boards:
        break







