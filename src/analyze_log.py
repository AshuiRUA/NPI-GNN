from os import name
from typing import Text
import networkx as nx
import copy
import numpy
import random
from openpyxl import Workbook
import torch
import sklearn

MCCS = []
ACCS = []
PRES = []
SENS = []
SPES = []

list_result = []

name_training = '1221_6'
line_start = 'Epoch: 035, testing dataset,'
for i in range(5):
    with open(f'result/{name_training}/log_{i}.txt') as f:
        for line in f.readlines():
            if line.startswith(line_start):
                list_result.append(line.strip())
                print(line)


for result in list_result:
    arr = result.split(',')
    ACC = float(arr[2].split(': ')[1])
    PRE = float(arr[3].split(': ')[1])
    SEN = float(arr[4].split(': ')[1])
    SPE = float(arr[5].split(': ')[1])
    MCC = float(arr[6].split(': ')[1])
    MCCS.append(MCC)
    ACCS.append(ACC)
    PRES.append(PRE)
    SENS.append(SEN)
    SPES.append(SPE)



print(f'{numpy.mean(MCCS)}')
print(f'{numpy.mean(ACCS)}')
print(f'{numpy.mean(PRES)}')
print(f'{numpy.mean(SENS)}')
print(f'{numpy.mean(SPES)}')