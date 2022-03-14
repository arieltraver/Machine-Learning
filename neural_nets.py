import helpers.py
import math, numpy


#takes numpy column vec, returns numpy column vec

def relu(z):
    return numpy.where(z>0, z[0], [0])
print(relu(numpy.array([[-2],[3],[0]])))

def drelu_dz(z):
    return numpy.where(z > 0, [1], [0])
