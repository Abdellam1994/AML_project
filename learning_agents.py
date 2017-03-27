# -*- coding: utf-8 -*-
import numpy as np

from constants import *
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
	Algorithm, using the least square method
	"""

	def __init__(self, number_players, _lambda=0.1 , gamma = 0.8, learning_rate = 0.1, epsilon = 0.3):
		self.revolution = 0
		self.state_length = 13
		self.n_actions = 4 * 13 + 1
		self.number_players = number_players
		self.learning_rate = learning_rate
		self.list_feat_state = [[0. for j in range(self.state_length * self.n_actions)] for i in xrange(self.n_actions)]

		self.state = [0. for i in range(self.state_length)]

		self.gamma = gamma
		self.beta = [np.random.uniform() for i in range(self.state_length * self.n_actions)]
		self.A = np.zeros((self.state_length * self.n_actions, self.state_length * self.n_actions))
		self.b = np.zeros((1, self.state_length * self.n_actions))

		self.Q_values = [np.random.uniform() for i in xrange(len(self.list_feat_state))]
		self.action = 0
		self.epsilon = epsilon
		self.rewards = []
		self._lambda = _lambda
		self.actual_feat_vector = self.list_feat_state[0]

    # There is also no need to update the state
	def updateState(self, last, hand, history, revolution, counter, heuristics) :

		self.revolution = revolution
		proba = probabilities(hand, history, revolution)
		self.state = np.concatenate((np.array([hand.count(elem) for elem in values]), proba), axis = 0)[:13]

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

	def LSTDQ_evaluation(self, reward):

		action = self.action
		feat_vector = np.array(self.list_feat_state[action])

		z_t = np.array(self.actual_feat_vector)

		self.A += np.dot(np.array([z_t]).T, np.array([z_t - self.gamma * feat_vector]))
		self.b += [z_t * reward]
		z_t = self._lambda * z_t + feat_vector
		b = self.b

		Beta = b.dot(np.linalg.pinv(self.A))
		self.actual_feat_vector = feat_vector

		return Beta

	# We update the revolution
	def update(self, reward, last, hand, history, revolution, moves, counter, heuristics):

		self.updateState(last, hand, history, revolution, counter, heuristics)
		for i in range(self.n_actions):
			for k in range(self.state_length):
				self.list_feat_state[i][i * self.state_length + k] = self.state[k]
		# Computing the different Q_values
		self.Q_values = []
		for i in range(self.n_actions):
			feat_vector_s_a = np.array(self.list_feat_state[i])
			q_s_a = np.dot(self.beta, feat_vector_s_a.T).tolist()
			self.Q_values.append(q_s_a)
		# Updating beta
		self.beta = self.LSTDQ_evaluation(reward)

		return None

# Agent UCB1

class UCB1_Agent:
	
	"""
	Class that implements an agent whose decisions are based the heuristics and UCB
	ALgorithm
	"""
	
	def __init__(self, number_players, gamma = 0.2, learning_rate = 0.1, epsilon = 0.3):
		self.revolution = 0
		self.number_players = number_players
		self.state_length = 4 + 13
		self.n_actions = 4 * 13 + 1

		# Rank of the learning agent
		self.state = [0 for i in range(self.state_length)]

		self.action = 0
		self.epsilon = epsilon
		self.rewards = []

		# Epsilon Greedy
		self.epsilon = epsilon

		self.visits_number = {}
		self.iteration_number = {}
		self.means = {}

    # There is also no need to update the state
	def updateState(self, last, hand, history, revolution, counter, heuristics) :

		self.revolution = revolution
		self.state = np.concatenate((np.array([hand.count(elem) for elem in values]), heuristics), axis = 0)

		return None

	def choose(self, pm) :
		
		# Computing the possible action
		possible_actions = [transform_state(move) for move in pm]
		
		if np.random.uniform() < self.epsilon :
			self.action = possible_actions[np.random.randint(len(possible_actions))]

		else :

			if str(self.state) not in self.means.keys():
				self.means[str(self.state)] = np.zeros(self.n_actions)
			if str(self.state) not in self.visits_number.keys():
				self.visits_number[str(self.state)] = np.zeros(self.n_actions)
			if str(self.state) not in self.iteration_number.keys():
				self.iteration_number[str(self.state)] = 1

			visit_number = self.visits_number[str(self.state)]
			it_number = self.iteration_number[str(self.state)]
			ucb_values = self.means[str(self.state)] + np.sqrt(2. * (np.log(it_number / visit_number)))

			ucb_values_possible = [ucb_values[i] if i in possible_actions else -100 for i in xrange(len(ucb_values))]
			self.action = np.argmax(ucb_values_possible)

			# Needs to transform the action into the appropriate format
		return transform_inverse(self.action)

    # We update the revolution
	def update(self, reward, last, hand, history, revolution, moves, counter, heuristics):

		self.updateState(last, hand, history, revolution, counter, heuristics)

		state = str(self.state)

		a = self.action

		if state not in self.means.keys():
			self.means[state] = np.zeros(self.n_actions)
		if state not in self.visits_number.keys():
			self.visits_number[state] = np.zeros(self.n_actions)

		if state not in self.iteration_number.keys():
			self.iteration_number[state] = 1
		self.means[state][a] *= (self.visits_number[state][a])
		self.means[state][a] += reward
		self.means[state] /= (self.visits_number[state][a] + 1)

		self.visits_number[state][a] += 1
		self.iteration_number[state] += 1
		return None