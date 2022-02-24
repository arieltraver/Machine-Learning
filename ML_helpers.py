import numpy, math

def MSE(data, theta, thetanaught=0, squared=0):
    total = 0
    for entry in data:
        test = entry[0][0] *  theta + thetanaught
        diff = entry[1] - test
        if(squared):
            diff = math.pow(diff, 2)
        total += diff
    return total / len(data)


arr = numpy.array([[[1],2],[[2],3],[[3],4]])
theta = 3
print(MSE(arr, theta, thetanaught=1, squared=1))
print("actual answer is 9")



