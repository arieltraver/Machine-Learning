import helpers
import math, numpy


#takes numpy column vec, returns numpy column vec

def softmax(a):
    e_to_a = numpy.exp(a)
    denom = e_to_a.sum(axis=0)
    return e_to_a / denom


def relu(z):
    return numpy.where(z>0, z[0], [0])

def drelu_dz(z):
    return numpy.where(z > 0, [1], [0])


print(relu(numpy.array([[-2],[3],[0]])))
print(softmax(numpy.array([[1],[1],[1]])))
