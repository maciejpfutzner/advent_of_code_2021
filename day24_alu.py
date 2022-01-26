#!/usr/bin/env python
# coding: utf-8

def parse_program(program):
    return [l.strip() for l in program.split('\n')]


with open('day24_monad_program.txt') as infile:
    monad = parse_program(infile.read())


monad


# # Part 1

def mock_input(num):
    num = str(num)
    if len(num) < 14:
        raise ValueError('Number too short')
    if '0' in num:
        raise ValueError('Number contains 0s')

    gen = (int(d) for d in num)
    def get_input():
        return next(gen)
    
    return get_input


class ALU():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.registers = dict(w=0, x=0, y=0, z=0)
        #self.verbose = False
        
    @staticmethod
    def get_instruction(line):
        op = line[:3]
        x = line[4]
        y = line[6:]
        return op, x, y
        
    def parse_operand(self, val):
        if val in 'wxyz':
            return self.registers[val]
        else:
            return int(val)
        
    def step(self, line):
        op, x, y = self.get_instruction(line)
        if op == 'inp':
            self.registers[x] = next(self.input_)
        elif op == 'add':
            self.registers[x] += self.parse_operand(y)
        elif op == 'mul':
            self.registers[x] *= self.parse_operand(y)
        elif op == 'div':
            self.registers[x] //= self.parse_operand(y)
        elif op == 'mod':
            self.registers[x] %= self.parse_operand(y)
        elif op == 'eql':
            self.registers[x] = int(self.registers[x] == self.parse_operand(y))
    
    def run(self, program, input_, verbose=False):
        self.reset()
        self.input_ = (int(i) for i in input_)
        for i, l in enumerate(program):
            self.step(l)
            if verbose:
                print(f'{i+1}: {l}')
                print(self.registers)
        return self.registers.copy()


alu = ALU()


binary = """inp w
add z w
mod z 2
div w 2
add y w
mod y 2
div w 2
add x w
mod x 2
div w 2
mod w 2"""

number = 4

program = parse_program(binary)
regs = alu.run(program, [number])

print(regs)
print(bin(number))


prog2 = """inp z
inp x
mul z 3
eql z x"""
input2 = [2, 1]

alu.run(parse_program(prog2), input2)


# ### Now try the monad program

top_number = int('9'*14)
top_number


alu.run(monad, str(top_number))


get_ipython().run_cell_magic('timeit', '', 'alu.run(monad, str(top_number))')

After line 30: w=d2, x=1, y=26, z=d1+15
After line 50: w=d3, x=1, y=0, z= ((d1+15)*26 + d2+12)*26
After line 70, asumming d4=d3+6: w=d4, x=0, y=0, z=(d1+15)*26 + d2+12
After line 90, same assumptions: w=d5, x=1, y=d5+15, z=(d1+15)*26 +d5+15
After line 120, if d7 = d6+1: w=d7, x=0, y=1, z=(d1+15)*26 +d5+15
After line 150: w=d9, x=d1+26, y=0, z=d1+15
# Conditions:
# - d3, d4: d3 = d4-6, so best d3=3, d4=9
# - d1, d8: d8 = d1-1, so best d1=9, d8=8
# - d2, d5: d5 = d2+5, so best d2=4, d5=9
# - d6, d7: d7 = d6+1, so best d6=8, d7=9
# - d9, d10: d10 = d9-5, so best d9=9, d10=4
# - d11, d14: d11 == d14, so both = 9
# - d12, d13: d13 = d12-4, so best d13=9, d12=5

num = 12345678912345
num = 94399898949959
regs = alu.run(monad, str(num), verbose=True)


regs['z']


# # Part 2
# 
# Conditions:
# - d1, d8: d1 = d8+1, so best d1=2, d8=1
# - d2, d5: d5 = d2+5, so best d2=1, d5=6
# - d3, d4: d4 = d3+6, so best d3=1, d4=7
# - d6, d7: d7 = d6+1, so best d6=1, d7=2
# - d9, d10: d9 = d10+5, so best d9=6, d10=1
# - d11, d14: d11 == d14, so both = 1
# - d12, d13: d12 = d13+4, so best d13=1, d12=5

num = 12345678901234
num = 21176121611511
regs = alu.run(monad, str(num), verbose=True)


regs['z']




