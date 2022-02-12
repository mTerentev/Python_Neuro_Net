from re import I
from NeuroNet import *
import numpy as np

def index_of_max(vector):
    index=0
    max=0
    i=0
    for a in vector:
        if a>max:
            index=i
            max=a
        i+=1
    return index

print(index_of_max(np.array([1,4,3,5,4])))