# -*- coding: utf-8 -*-

import operator as op

from constants import *

import numpy as np

from utils import probabilities, transform_state, transform_inverse


# Function to go back and forth from the neural network's representation of the
# Q-values to the actual Q-values through an affine transformation
def to_NN(Q, fr=[10, 5, 0, -5, -10]):
    nn = Q * (np.max(fr) + 13 - np.min(fr))
    nn += np.min(fr)
    return nn
		

# Neural network class to use for approximating the Q-values
class NN:
    def __init__(self, Length, learning_rate):
		self.coefs = [np.random.normal(0., 1., (Length[i + 1], Length[i])) for i in xrange(len(Length) - 1)]
		self.learning_rate = learning_rate
		
    def train(self, x, t):
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
        self.coefs = [self.coefs[i] - self.learning_rate * grad[i] for i in xrange(len(self.coefs))]
        return inter[-1]

    def predict(self, x):
        inter = x
								
        # Forward propagation
        for i in xrange(len(self.coefs)):
            v = self.coefs[i].dot(inter)
            inter = 1. / (1 + np.exp(-v))

        return inter


# Agent CNN

class NNQL_Agent:
	
	"""
	Class that implements an agent whose decisions are based on the Q-Learning
	Algorithm, using a neural network as an approximation of the Qvalue
	"""
	
	def __init__(self, number_players, gamma = 0.2, learning_rate = 0.1, epsilon = 0.3):
		self.revolution = 0
		self.number_players = number_players
		self.learning_rate = learning_rate
		self.list_NN = [NN([5 * 13 + 13 + 1, 1], self.learning_rate) for i in xrange(4 * 13 + 1)]
		#Rank of the learning agent
		self.state = 0
		# Parameter of learning for Neural Networks
		self.gamma = gamma
		self.Q_values = [np.random.uniform() for i in xrange(len(self.list_NN))]
		self.action = 0
		self.epsilon = epsilon
		self.rewards = []
		
		
    # There is also no need to update the state
	def updateState(self, last, hand, history, revolution, counter, heuristics) :
		self.revolution = revolution
		proba = probabilities(hand, history, revolution)
		self.state = np.concatenate((np.array([hand.count(elem) for elem in values]), proba), axis = 0)
		return None

	def choose(self, pm) :
		
		# Computing the possible action
		possible_actions = [transform_state(move) for move in pm]
		
		if np.random.uniform() < self.epsilon :
			self.action = possible_actions[np.random.randint(len(possible_actions))]
		
		else :
			
			# The Q values corresponding to the possible moves
			Q_values_possible = [self.Q_values[i] if i in possible_actions else -100 for i in xrange(len(self.Q_values))]
			self.action = np.argmax(Q_values_possible)
			# Needs to transform the action into the appropriate format
		return transform_inverse(self.action)

    # We update the revolution
	def update(self, reward, last, hand, history, revolution, moves, counter, heuristics):
		self.updateState(last, hand, history, revolution, counter, heuristics)
		# Computing the different Q_values
		self.Q_values = [NN_.predict(self.state)[0] for NN_ in self.list_NN]
		self.list_NN[self.action].train(self.state, reward + self.gamma * self.Q_values[self.action])
		return None
		
		
		
# Agent LSTD

class LSTD_Agent:
	
	"""
	Class that implements an agent whose decisions are based on the Q-Learning
	Algorithm, using a neural network as an approximation of the Qvalue
	"""
	
	def __init__(self, number_players, gamma = 0.2, learning_rate = 0.1, epsilon = 0.3):
		self.revolution = 0
		self.number_players = number_players
		self.learning_rate = learning_rate
		self.list_NN = [NN([5 * 13 + 13 + 1, 1], self.learning_rate) for i in xrange(4 * 13 + 1)]
		#Rank of the learning agent
		self.state = 0
		# Parameter of learning for Neural Networks
		self.gamma = gamma
		self.Q_values = [np.random.uniform() for i in xrange(len(self.list_NN))]
		self.action = 0
		self.epsilon = epsilon
		self.rewards = []
		
		
    # There is also no need to update the state
	def updateState(self, last, hand, history, revolution, counter, heuristics) :
		self.revolution = revolution
		proba = probabilities(hand, history, revolution)
		self.state = np.concatenate((np.array([hand.count(elem) for elem in values]), proba), axis = 0)
		return None

	def choose(self, pm) :
		
		# Computing the possible action
		possible_actions = [transform_state(move) for move in pm]
		
		if np.random.uniform() < self.epsilon :
			self.action = possible_actions[np.random.randint(len(possible_actions))]
		
		else :
			
			# The Q values corresponding to the possible moves
			Q_values_possible = [self.Q_values[i] if i in possible_actions else -100 for i in xrange(len(self.Q_values))]
			self.action = np.argmax(Q_values_possible)
			# Needs to transform the action into the appropriate format
		return transform_inverse(self.action)

    # We update the revolution
	def update(self, reward, last, hand, history, revolution, moves, counter, heuristics):
		self.updateState(last, hand, history, revolution, counter, heuristics)
		# Computing the different Q_values
		self.Q_values = [NN_.predict(self.state)[0] for NN_ in self.list_NN]
		self.list_NN[self.action].train(self.state, reward + self.gamma * self.Q_values[self.action])
		return None
		
		
# Agent UCB1

class UCB1_Agent:
	
	"""
	Class that implements an agent whose decisions are based on the Q-Learning
	Algorithm, using a neural network as an approximation of the Qvalue
	"""
	
	def __init__(self, number_players, gamma = 0.2, learning_rate = 0.1, epsilon = 0.3):
		self.revolution = 0
		self.number_players = number_players
		self.learning_rate = learning_rate
		self.list_NN = [NN([5 * 13 + 13 + 1, 1], self.learning_rate) for i in xrange(4 * 13 + 1)]
		#Rank of the learning agent
		self.state = 0
		# Parameter of learning for Neural Networks
		self.gamma = gamma
		self.Q_values = [np.random.uniform() for i in xrange(len(self.list_NN))]
		self.action = 0
		self.epsilon = epsilon
		self.rewards = []
		
		
    # There is also no need to update the state
	def updateState(self, last, hand, history, revolution, counter, heuristics) :
		self.revolution = revolution
		proba = probabilities(hand, history, revolution)
		self.state = np.concatenate((np.array([hand.count(elem) for elem in values]), proba), axis = 0)
		return None

	def choose(self, pm) :
		
		# Computing the possible action
		possible_actions = [transform_state(move) for move in pm]
		
		if np.random.uniform() < self.epsilon :
			self.action = possible_actions[np.random.randint(len(possible_actions))]
		
		else :
			
			# The Q values corresponding to the possible moves
			Q_values_possible = [self.Q_values[i] if i in possible_actions else -100 for i in xrange(len(self.Q_values))]
			self.action = np.argmax(Q_values_possible)
			# Needs to transform the action into the appropriate format
		return transform_inverse(self.action)

    # We update the revolution
	def update(self, reward, last, hand, history, revolution, moves, counter, heuristics):
		self.updateState(last, hand, history, revolution, counter, heuristics)
		# Computing the different Q_values
		self.Q_values = [NN_.predict(self.state)[0] for NN_ in self.list_NN]
		self.list_NN[self.action].train(self.state, reward + self.gamma * self.Q_values[self.action])
		return None