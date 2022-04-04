import numpy as np

#simple representation of a neural net using standard gradient descent
#starter code from MIT


class Module:
    def sgd_step(self, lrate): pass  # For modules w/o weights

class Linear(Module):
    def __init__(self, m, n):
        # initializes the weights randomly and offsets as 0
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        # store the input matrix for future use
        self.A = A   # (m x b)  Hint: make sure you understand what b stands for
        wt = np.transpose(self.W)
        return wt @ self.A + self.W0  # Your code (n x b)

    def backward(self, dLdZ):
        # dLdZ is (n x b), uses stored self.A
        # store the derivatives for use in sgd_step and returd dLdA
        self.dLdW = self.A @ np.transpose(dLdZ)    # Your code
        self.dLdW0 = dLdZ@np.ones((dLdZ.shape[1], 1))#nxb
        return self.W @ dLdZ           # Your code: return dLdA (m x b)

    def sgd_step(self, lrate):  # Gradient descent step
        sub = lrate * self.dLdW
        self.W = self.W - sub         # Your code
        sub0 = lrate * self.dLdW0
        self.W0 = self.W0 - sub0      # Your code

class Tanh(Module):            # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z)            # Your code
        return self.A

    def backward(self, dLdA):    # Uses stored self.A
        deriv = np.ones(self.A.shape) - self.A * self.A #gives dA/dZ = 1 - A(z)^2
        return dLdA * deriv #dLdZ = dLdA * dAdZ
