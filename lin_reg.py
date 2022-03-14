import numpy, math, random

#arrays should take the form [[x1],y1],[[x2],y2],[[x3],y3]] 
#this is bc our input/output is 1-d right now but maybe later will be more
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

#lil bit of functional programming for you <3
#inputs is the x,y entries that you have
#gradient func takes three parameters: x, y (from inputs), and theta
def gradient_step(theta, step_size, gradient_func, inputs):
    total = 0
    trials = numpy.ones(len(inputs))
    index = 0
    for entry in inputs:
        trials[index] = gradient_func(entry[0][0], entry[1], theta)
        index += 1
    for trial in trials:
        total += trial
    total /= len(trials) #average
    total *= step_size
    return total

def sample_gradient(x, y, theta):
    return theta * x * x - theta * x * y

print(gradient_step(3, 0.2, sample_gradient, arr))


#random stochastic gradient step
def rand_stoch_step(theta, gradient_func, inputs, step_size):
    #select a random piece of data
    trial = random.randrange(len(inputs) - 1)
    #instead of summing up all of them, just do one
    val = gradient_func(inputs[trial][0][0], inputs[trial][1], theta)
    val *= step_size
    return val


theta = -1/9
step = 0.2
data = [[[1,1]],[[2,2]]]




