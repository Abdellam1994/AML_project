# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:54:02 2017

@author: Abdellah
"""




""" In this file, we declare some function useful for the game.
For instance a function that renders the two best cards of some player."""

import operator as op

from constantes import *



"""This function renders the final reward of the game according to the number of players,
it takes as input :
- n : the number of players
- reward_array : an array containing the rewards for the different positions (president, 
			    vice-president, people, vice-trou, trou).
"""																																																																																
def game_reward(n, reward_array = [1000, 500, 0, -500, -1000]):
	# Dictionary of rewards
    Reward = {1 : reward_array[0], 2 : reward_array[1], n - 1 : reward_array[3], n : reward_array[4]}
    
	# Plugging the score for the people to the players in position (2 : n-2)
	# (if there are more than 4 players)
    if n >= 4 :
		for k in xrange(3, n - 1):
			Reward[k] = reward_array[2]
    return Reward


""" This function returns a list of the n best cards of a hand for the exchange 
of cards step in the game. It takes as parameters :
- n : number of cards.
- cards : the hand of the player.
- revolution : state of the game (if there is a revolution).
"""
def find_best(n, cards, revolution = 0):
	# Initializing the list of cards
    best_cards = []
	#  storing the rank according to the state of the game (revolution = 0 or 1)
    rank_cards = [ranks[revolution][card] for card in cards]				
    for k in xrange(n):
		# Choosing the best card			
        i = np.argmax(L)
        best_cards.append(rev_ranks[revo][rank_cards[i]])
        rank_cards[i] = 0
    return best_cards


# Function used to get the n worst cards out of a hand
def find_worst(n, cards, revolution =0):
    wc = []
    L = [ranks[revo][card] for card in cards]
    for k in xrange(n):
        i = np.argmin(L)
        wc.append(rev_ranks[revo][L[i]])
        L[i] = 13
    return wc
				
				


def comb(p, n):
    r = min(p, n - p)
    if r == 0:
        return 1
    numer = reduce(op.mul, xrange(n, n - r, -1))
    denom = reduce(op.mul, xrange(1, r + 1))
    return numer // denom


# Function to get the probabilities of each player having a certain number of a card
# for every card
def compute_probabilities(hand, history, order, pj, revo):
    probas = []
    players_left = 0
    for pl in xrange(len(order)):
        if history.left[pl] != 0:
            players_left += 1
    if players_left > 1:
        for k in order:
            if k != pj:
                hist = history.players[k]
                for card in ranks[0].keys():
                    m = hand[card]
                    p = 4 - history.left[card]
                    cj = hist[card]
                    lim = min(4 - m - p, cj)
                    for i in xrange(1 + lim):
                        probas.append(comb(i, lim) * (1. / players_left) ** i * (1 - 1. / players_left) ** (lim - i))
                    if lim < 4:
                        for i in xrange(1 + lim, 5):
                            probas.append(0.)
    else:
        for k in order:
            if k != pj:
                for card in ranks[0].keys():
                    for i in xrange(5):
                        probas.append(0.)
    probas.append(players_left)
    probas.append(revo)
    return np.array(probas)
				
###################################################################################################
###################################### Unit tests #################################################	
###################################################################################################
				
# Unit tests in order to check some functions
				
import unittest

class TestFunctions(unittest.TestCase):

	def test_finalreward(self):
		self.assertEqual(game_reward(5, reward_array = [1000,500,0,-500,-1000]), {1 : 1000, 2 : 500, 3 : 0, 4 :-500 , 5 : -1000})
		self.assertEqual(game_reward(6, reward_array = [1000,500,0,-500,-800]), {1 : 1000, 2 : 500, 3 : 0, 4 : 0, 5 :-500 , 6 : -800})
	
	def test_find_best(self):
		# TO DO
		return None						
												
if __name__ == '__main__':
    unittest.main()
				
