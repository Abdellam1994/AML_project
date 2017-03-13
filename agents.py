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


class minAgent:
    # Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        return

    # There is also no need to update the state
    def updateState(self, last, cards, history, revo, counter):
        self.revo = revo

    # Here we choose the simple strategy of throwing the lowest value possible with the highest number possible
    def choose(self, pm):
        if len(pm) > 1:
            values = np.array([[ranks[self.revo][move[0]], move[1]] for move in pm if move != (0, 0)])
            card = np.min(values[:, 0])
            inds = (values[:, 0] == card)
            move = (rev_ranks[self.revo][card], np.max(values[inds, 1]))
            return move
        else:
            return (0, 0)

    # We update the revolution
    def update(self, reward, last, cards, history, revo, moves, counter):
        self.updateState(last, cards, history, revo, counter)
        return


class maxAgent:
    # Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        return

    # There is also no need to update the state
    def updateState(self, last, cards, history, revo, counter):
        self.revo = revo

    # Here we choose the simple strategy of throwing the lowest value possible with the highest number possible
    def choose(self, pm):
        if len(pm) > 1:
            values = np.array([[ranks[self.revo][move[0]], move[1]] for move in pm if move != (0, 0)])
            card = np.max(values[:, 0])
            inds = (values[:, 0] == card)
            move = (rev_ranks[self.revo][card], np.max(values[inds, 1]))
            return move
        else:
            return (0, 0)


class RealPlayer:
    def __init__(self):
        self.last = (0, 0)
        self.revo = 0
        return

    def updateState(self, last, cards, history, revo, counter):
        self.revo = revo
        self.last = last
        return

    def choose(self, pm):
        print("Last card thrown : " + str(self.last))
        if len(pm) != 1:
            pcn = {}
            for move in pm:
                if move[0] != 0:
                    if move[0] in pcn.keys():
                        pcn[move[0]].append(move[1])
                    else:
                        pcn[move[0]] = [move[1]]
                else:
                    pcn['0'] = [0]
            print(pcn)
            move = []
            while len(move) == 0:
                card = str(raw_input("Choose a card\n"))
                if card in pcn.keys():
                    move.append(card)
            while len(move) == 1:
                if self.last == (0, 0):
                    number = input("Choose a number of cards\n")
                    if int(number) in pcn[move[0]]:
                        move.append(number)
                    else:
                        print("You can't throw this number of cards")
                else:
                    move.append(pcn[move[0]][0])
            return tuple(move)
        else:
            return (0, 0)

    # We update the revolution
    def update(self, reward, last, cards, history, revo, moves, counter):
        self.updateState(last, cards, history, revo, counter)
        return


class minMaxAgent:
    # Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        self.minMax = 0
        return

    # There is also no need to update the state
    def updateState(self, last, cards, history, revo, counter):
        self.revo = revo

    # Here we choose the simple strategy of alternating between the lowest and largest values
    def choose(self, pm):
        if len(pm) > 1:
            if self.minMax == 0:
                values = np.array([[ranks[self.revo][move[0]], move[1]] for move in pm if move != (0, 0)])
                card = np.min(values[:, 0])
                inds = (values[:, 0] == card)
                move = (rev_ranks[self.revo][card], np.max(values[inds, 1]))
                return move
            else:
                values = np.array([[ranks[self.revo][move[0]], move[1]] for move in pm if move != (0, 0)])
                card = np.max(values[:, 0])
                inds = (values[:, 0] == card)
                move = (rev_ranks[self.revo][card], np.max(values[inds, 1]))
                return move
        else:
            return (0, 0)

    # We update the revolution
    def update(self, reward, last, cards, history, revo, moves, counter):
        self.minMax = 1 - self.minMax
        self.updateState(last, cards, history, revo, counter)
        return