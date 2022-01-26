#!/usr/bin/env python
# coding: utf-8

from io import StringIO
from functools import reduce


data = '0052E4A00905271049796FB8872A0D25B9FB746893847236200B4F0BCE5194401C9B9E3F9C63992C8931A65A1CCC0D222100511A00BCBA647D98BE29A397005E55064A9DFEEC86600BD002AF2343A91A1CCE773C26600D126B69D15A6793BFCE2775D9E4A9002AB86339B5F9AB411A15CCAF10055B3EFFC00BCCE730112FA6620076268CE5CDA1FCEB69005A3800D24F4DB66E53F074F811802729733E0040E5C5E5C5C8015F9613937B83F23B278724068018014A00588014005519801EC04B220116CC0402000EAEC03519801A402B30801A802138801400170A0046A800C10001AB37FD8EB805D1C266963E95A4D1A5FF9719FEF7FDB4FB2DB29008CD2BAFA3D005CD31EB4EF2EBE4F4235DF78C66009E80293AE9310D3FCBFBCA440144580273BAEE17E55B66508803C2E0087E630F72BCD5E71B32CCFBBE2800017A2C2803D272BCBCD12BD599BC874B939004B5400964AE84A6C1E7538004CD300623AC6C882600E4328F710CC01C82D1B228980292ECD600B48E0526E506F700760CCC468012E68402324F9668028200C41E8A30E00010D8B11E62F98029801AB88039116344340004323EC48873233E72A36402504CB75006EA00084C7B895198001098D91AE2190065933AA6EB41AD0042626A93135681A400804CB54C0318032200E47B8F71C0001098810D61D8002111B228468000E5269324AD1ECF7C519B86309F35A46200A1660A280150968A4CB45365A03F3DDBAE980233407E00A80021719A1B4181006E1547D87C6008E0043337EC434C32BDE487A4AE08800D34BC3DEA974F35C20100BE723F1197F59E662FDB45824AA1D2DDCDFA2D29EBB69005072E5F2EDF3C0B244F30E0600AE00203229D229B342CC007EC95F5D6E200202615D000FB92CE7A7A402354EE0DAC0141007E20C5E87A200F4318EB0C'
data


example1 = 'D2FE28'
example2 = '38006F45291200'


# # Part 1

def hex_to_bin(data):
    data_bin = ''.join([f'{int(d, base=16):04b}' for d in data])
    return data_bin


def read_literal(stream):
    read_counter = 0
    finished = False
    digits = ''
    while not finished:
        chunk = stream.read(5)
        read_counter += 5
        
        if chunk[0] == '0':
            finished = True
        digits += chunk[1:]
        #print(digits)
    value = int(digits, base=2)
    return value, read_counter


def read_op(pid, stream, versions):
    print('operator', pid)
    indicator = stream.read(1)
    read_counter = 1
    
    vals = []
    if indicator == '0':
        length = int(stream.read(15), base=2)
        read_counter += 15
        print(f'{length=}')
        
        sub_length = 0
        while sub_length < length:
            val, n = read_packet(stream, versions)
            sub_length += n
            vals.append(val)
        read_counter += sub_length
        
    else:
        number = int(stream.read(11), base=2)
        read_counter += 11
        print(f'{number=}')
        for i in range(number):
            val, n = read_packet(stream, versions)
            read_counter += n
            vals.append(val)
            
    # apply operation
    value = op_map[pid](vals)
    #print(f'operation {pid} on values {vals} gives {value}')
        
    return value, read_counter


def read_packet(stream, versions):
    version = int(stream.read(3), base=2)
    pid = int(stream.read(3), base=2)
    read_counter = 6
    print(f'{version=}')
    versions.append(version)
    
    if pid == 4:
        # literal value
        value, count = read_literal(stream)
        read_counter += count
    else:
        # operator
        value, count = read_op(pid, stream, versions)
        read_counter += count
        
    print(f'{value=}')
    return value, read_counter


def decode(data):
    versions = []
    stream = StringIO(hex_to_bin(data))
    return read_packet(stream, versions), versions


decode(example1)


decode('38006F45291200')


decode('EE00D40C823060')


decode('8A004A801A8002F478')


decode('620080001611562C8802118E34')


returns, versions = decode(data)


# # Part 2

op_map = {
    0: sum,
    1: lambda x: reduce(int.__mul__, x),
    2: min,
    3: max,
    5: lambda x: int(x[0] > x[1]),
    6: lambda x: int(x[0] < x[1]),
    7: lambda x: int(x[0] == x[1]),
}


code = 1
op_map[code]((6,9))


(val, n), versions = decode('C200B40A82')
print('final value', val)


(val, n), versions = decode('04005AC33890')
print('final value', val)


(val, n), versions = decode('04005AC33890')
print('final value', val)


(val, n), versions = decode('9C0141080250320F1802104A08')
print('final value', val)


(val, n), versions = decode(data)
print('final value', val)


val




