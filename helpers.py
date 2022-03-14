import math, numpy

def make_col(length):
    try:
        vectr = numpy.zeros(shape=(length, 1)) #column vector
        return vectr
    except:
        print("wrong input type") 
