# -*- coding: utf-8 -*-

import operator as op

from constants import *


# Function to go back and forth from the neural network's representation of the
# Q-values to the actual Q-values through an affine transformation
def to_NN(Q, fr=[10, 5, 0, -5, -10]):
    nn = Q * (np.max(fr) + 13 - np.min(fr))
    nn += np.min(fr)
    return nn
		


# Neural network class to use for approximating the Q-values
class NN:
    def __init__(self, L):
        self.coefs = [np.random.normal(0., 1., (L[i + 1], L[i])) for i in xrange(len(L) - 1)]

    def train(self, x, t, lr):
        inter = [x]
        grad = [np.zeros(K.shape) for K in self.coefs]

        # Forward propagation
        for i in xrange(len(self.coefs)):
            v = self.coefs[i].dot(inter[-1])
            inter.append(1. / (1 + np.exp(-v)))

        # Backward propagation
        forward = inter[-1]
        delta = forward * (1 - forward) * (t - forward)
        grad[-1] = delta * inter[-2]
        if len(self.coefs) >= 2:
            for i in range(-2, -len(self.coefs) - 1, -1):
                delta = inter[i] * (1 - inter[i]) * (np.transpose(self.coefs[i + 1]).dot(delta))
                grad[i] = delta.reshape([delta.shape[0], 1]).dot(inter[i - 1].reshape(1, inter[i - 1].shape[0]))

        # Updating the weights
        self.coefs = [self.coefs[i] - lr * grad[i] for i in xrange(len(self.coefs))]
        return inter[-1]

    def predict(self, x):
        inter = x

        # Forward propagation
        for i in xrange(len(self.coefs)):
            v = self.coefs[i].dot(inter)
            inter = 1. / (1 + np.exp(-v))

        return inter
