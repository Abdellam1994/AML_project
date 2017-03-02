# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:54:02 2017

@author: Abdellah
"""




""" In this file, we declare some function useful for the game.
For instance a function that renders the two best cards of some player."""

import operator as op

from constantes import *



																																																																														
def game_reward(n, reward_array = [1000, 500, 0, -500, -1000]):
	
	"""
	This function renders the final reward of the game according to the number of players,
	it takes as input :
	- n : the number of players
	- reward_array : an array containing the rewards for the different positions (president, 
				    vice-president, people, vice-trou, trou).
	"""
		
	# Dictionary of rewards
	Reward = {1 : reward_array[0], 2 : reward_array[1], n - 1 : reward_array[3], n : reward_array[4]}
    
	# Plugging the score for the people to the players in position (2 : n-2)
	# (if there are more than 4 players)
	if n >= 4 :
		for k in xrange(3, n - 1):
			Reward[k] = reward_array[2]
	return Reward



def find_best(n, cards, revolution = 0):
	
	""" 
	This function returns a list of the n best cards of a hand for the exchange 
	of cards step in the game. It takes as parameters :
	- n : number of cards.
	- cards : the hand of the player.
	- revolution : state of the game (if there is a revolution).
	"""
	
	# Initializing the list of cards
	best_cards = []
	#  storing the rank according to the state of the game (revolution = 0 or 1)
	rank_cards = [ranks[revolution][card] for card in cards]				
	for k in xrange(n):
		# Choosing the best card			
        i = np.argmax(rank_cards)
        best_cards.append(rev_ranks[revolution][rank_cards[i]])
		# Removing the best card
        rank_cards.remove(rank_cards[i])
    return best_cards


def find_worst(n, cards, revolution = 0):
	
	""" 
	This function returns a list of the n worst cards of a hand for the exchange 
	of cards step in the game. It takes as parameters :
	- n : number of cards.
	- cards : the hand of the player.
	- revolution : state of the game (if there is a revolution).
	"""
	
	# Initializing the list of cards
    worst_cards = []
    #  storing the rank according to the state of the game (revolution = 0 or 1)
    rank_cards = [ranks[revolution][card] for card in cards]
    for k in xrange(n):
        # Choosing the best card			
        i = np.argmin(rank_cards)
        worst_cards.append(rev_ranks[revolution][rank_cards[i]])
        # Removing the worst card
        rank_cards.remove(rank_cards[i])
    return worst_cards
				


def binomial_coeff(p, n):
	
	""" 
	This function computes the binomial coefficients, takes as parameters :
	- p : an integer
	- n : an integer
	"""
	
	# Optimizing the computation according to which of p and n - p is the smallest
    min_coeff = min(p, n - p)
    if min_coeff == 0:
        return 1
	# Computing (n ! / (n - p)!) or (n ! / (p!)) according to the value of min(p, n - p)
    numerator = reduce(op.mul, xrange(n, n - min_coeff, -1))
	# Computing (n - p)! or p! according to the value of min(p, n - p)
    denominator = reduce(op.mul, xrange(1, min_coeff + 1))	
	# Returning the eucledian division (binomial coefficients are integers).
    return numerator // denominator



####################################################################
############# TO DO : Continue changing the function ############### 
####################################################################


def probabilities(hand, history, order, pj, revolution):
	
	""" 
	This function computes the probabilities for each player to have a certain amount of a 
	certain card given the history and our hand. It will compose our state of space.
	It takes as parameters :
	- hand : list of cards of the player (the hand).
	- history : the history of the game. (class)
	- order : the order to know which player had played which card.
	- pj :
	- revolution : boolean variable, state of the game, either there was a revolution or not.
	
	"""
	
	# Storing the probabilities
    probabilities = []
				
	# Computing the number of remaining players
    remaining_players = sum(player > 0 for player in history.nb_cards_player.values())

												
    if players_left > 1:
					
        for k in order :
									
            if k != pj :
													
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
    return np.array(probabilities)
				
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
		hand = ['A', 'Q', '3', 'A']	
		self.assertEqual(find_best(2, hand), ['A','A'])	
		self.assertEqual(find_best(2, hand, revolution = 1), ['3','Q'])

	def test_find_worst(self):
		hand = ['A', 'Q', '3', 'A', '4','2']	
		self.assertEqual(find_worst(2, hand), ['3','4'])	
		self.assertEqual(find_worst(2, hand, revolution = 1), ['2','A'])			
											
if __name__ == '__main__':
    unittest.main()
				
