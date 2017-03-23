import numpy as np
import time

from constants import ranks, rewards, statuses, deck, rev_ranks, rank_max
from utils import find_best, find_worst, game_reward


"""

	This file contains the three main classes of the game :
	- history : in order to keep the history of the game.
	- player : the player that plays with an agent.
	-  game : the class that handles the game.

"""

class history:

	""" Class in order to keep the history of the game for each player """

	def __init__(self, n) :
		
		"""
		Constructor method : is initialized with the number of players.
		"""

		# History for each player, cards that have been played.
		self.players = [{card : 0 for card in ranks[0].keys()} for k in xrange(n)]
		# Remaining cards in the game
		self.remaining_cards = {card : 4 for card in ranks[0].keys()}
		# Number of cards left for each player
		self.nb_cards_player = dict({player : 52 // n + 1 for player in xrange(n) if (player  < 52 % n)}, **{player : 52 // n for player in xrange(n) if(player >= 52 % n)})

	def our_hand(self, hand):
		
		"""
		Function that substract from the remaining cards (history) the cards of the player.
		
		Parameters
		----------

		- hand : hand of the player, list of lists [['A',3], ['3',2]] for instance
		"""
		for card in hand:
			self.remaining_cards[card[0]] -= card[1]
		return None

	def update(self, player, move):
		
		"""
		Function that updates the history.
		
		Parameters
		----------

		- player : the player that played the game.
		- move : the card and number of cards played. move = ['A', 3] for instance.
		"""
		# If the move does not mean passing
		if move[1] != 0:
            # Updating the cards played by the player
			self.players[player][move[0]] += move[1]
            # Updating the number of cards of the value played
			self.remaining_cards[move[0]] -= move[1]
            # Updating the cards left for the player
			self.nb_cards_player[player] -= move[1]
		return None


class player:
	
    """ 
	
	Class representing a player with a hand, and an agent to make decisions 
				
	"""
	
    def __init__(self, Agent, cards):
        # Defining an agent for the player
		self.agent = Agent
		self.cards = cards
		# np.array([0 for k in xrange(15)] + hand)
		# Status of the player (president, people ...)
		self.status = 'People'
		# is the player playing ?
		self.out = 0
		# initilializing the list of rewards
		self.rewards = []

    def possible_moves(self, last, revolution=0, pass_=False):

		"""
		Method that determines the possible moves for the player.

		Parameters
		----------

		last : tuple containing (card_chosen, number_of_cards)
		revolution : binarary variable
		pass_ : if the player has passed

		"""

		# If the agent is actually a realer player : show the cards
		if str(self.agent).split('.')[1].split(' ')[0] ==  "RealPlayer" :
			print(self.cards)

		# Default possible move
		possible_moves_ = [(0, 0)]

		# Only not playing option
		if pass_:
			return possible_moves_

        # If the player doesn't initiate the turn
		if last[1] != 0:

            # Get the value of the last card(s)
			v = ranks[revolution][last[0]]

			# Get the value of the cards in hand
			w = [ranks[revolution][card] for card in self.cards]

			# Get the playable cards with their cardinality
			L = {n: np.sum([n == value for value in w]) for n in xrange(1 + v, 1 + rank_max)}

			# If the player is trou or vice-trou, the equality right activates
			if self.status in statuses[3:]:
				L[v] = np.sum([v == value for value in w])

			# Adding the possible moves : play the same number of cards as the last play
			for n in L.keys() :
				if L[n] >= last[1]:
					possible_moves_.append((rev_ranks[self.agent.revo][n], last[1]))

        # If the player initiates the turn
		else :
            # Get the value of the cards in hand
			w = [ranks[revolution][card] for card in self.cards]
			# Get the cardinality of the cards at hand
			L = {n : np.sum([n == value for value in w]) for n in set(w)}
			# Adding the possibe moves : play any possible number of any card at hand
			for n in L.keys():
				for k in xrange(L[n]):
					possible_moves_.append((rev_ranks[self.agent.revo][n], 1+k))
		return possible_moves_
    
	
    def play(self, move):
					
		"""
		Method that removes the card played.

		Parameters
		----------

		move : move done. (tuple)

		"""
					
		if move == (0, 0):
			return None
		else:
			for k in xrange(move[1]):
				self.cards.remove(move[0])
			return None


    def choose(self, last, revolution, history, counter, pass_) :
					
		"""
		Method that chooses an action according to the agent used.

		Parameters
		----------

		last : tuple containing (card_chosen, number_of_cards).
		revolution : binarary variable.
		history : historic of the game (class).
		counter : counter used by certain agents (Q-Learning, Bandits).
		pass_ : binary variable.
		"""
		
		# Last,cards,history,revo,counter
		self.agent.updateState(last, self.cards, history, revolution, counter)
		
		return self.agent.choose(self.possible_moves(last, revolution, pass_))

	# Method to update the agent
    def update(self, reward, last, history, revolution, counter):
					
		"""
		Method that updates the agent.

		Parameters
		----------

		reward : the reward obtained.
		last : tuple containing (card_chosen, number_of_cards).
		revolution : binarary variable.
		history : historic of the game (class).
		counter : counter used by certain agents (Q-Learning, Bandits).
		"""
		
		# Storing the rewards obtained
		self.rewards.append(reward)
		self.agent.update(reward, last, self.cards, history, revolution, self.possible_moves(last, revolution), counter)
		return None

# Game class containing players each with their own agent they use to make decisions


class GAME:
	
	""" Class that handles the game, contains the players with their agents  """

	def __init__(self, agents, number_player = 4, final = [10, 5, 0, -5, -10], verbose = True):
		
		"""
		Constructor method : is initialized with the number of players.
		
		Parameters		
		----------
		
		last : tuple containing (card_chosen, number_of_cards).
		final : rewards for each players.
		"""
		
        # The stack is empty
		self.last = (0, 0)

		# Setting the rewards
		self.final = game_reward(number_player, final)

        # Shuffling the cards
		np.random.shuffle(deck)

        # Counting the cards
		q = 52 // number_player
		r = 52 % number_player
		self.players = []

        # Creating the players and distributing the cards
		if r == 0:
			self.players += [player(agents[k], list(deck[k * q: (k+1) * q])) for k in xrange(number_player)]
		else :
			self.players += [player(agents[k], list(deck[k * (q+1): (k+1) * (q+1)])) for k in xrange(r)]
			self.players += [player(agents[k], list(deck[r * (q+1) + k * q: r * (q+1) + (k+1) * q])) for k in xrange(number_player-r)]

        # Setting the order at which the players play
		self.order = range(number_player)

        # No revolution at the begining of the game
		self.revolution = 0

        # The first player starts
		self.actual_player = 0

        # History of cards played
		self.history = history(number_player)

        # Counter of people that left the game
		self.counter = 0

        # Counting the passes for the initiative transfer
		self.passes = 0
		
		# The heuristics variables coded (list of variables)
		self.heuristics = [0 for i in xrange(number_player)]
								
		# Verbose (printing or not)
		self.verbose = verbose


	def reset(self):
					
		"""
		Method that resets the game.
		"""
		
        # Resetting the game
		self.last = (0, 0)
		np.random.shuffle(deck)
		number_of_players = len(self.players)

		q = 52 // number_of_players
		r = 52 % number_of_players
		if r != 0:
			for k in xrange(r):
				self.players[k].cards = list(deck[(q+1) * k: (q+1) * (k+1)])
			for k in xrange(number_of_players - r):
				self.players[r+k].cards = list(deck[(q+1) * r + q * k: (q+1) * r + q * (k+1)])
		else:
			for k in xrange(number_of_players):
				self.players[k].cards = list(deck[q * k: q * (k+1)])
		self.revolution = 0
		self.counter = 0
		# Doing the exchange of cards for the two highest and two lowest ranked
		ind = [0 for k in xrange(4)]
		exchanges = [0 for k in xrange(4)]
		# Choosing the cards to be exchanged
		for i in xrange(number_of_players):
			if self.players[i].status == 'Trou':
				ind[0] = i
				exchanges[0] = find_best(2, self.players[i].cards)
				print("Player "+str(i)+" is the trou.")
			if self.players[i].status == 'Vice-trou':
				ind[1] = i
				exchanges[1] = find_best(1, self.players[i].cards)
				print("Player "+str(i)+" is the vice-trou.")
			if self.players[i].status == 'Vice-president':
				ind[2] = i
				exchanges[2] = find_worst(1, self.players[i].cards)
				print("Player "+str(i)+" is the vice-president.")
			if self.players[i].status == 'President':
				ind[3] = i
				exchanges[3] = find_worst(2, self.players[i].cards)
				print("Player "+str(i)+" is the president.")
		# If we're not at the begining of the game, we perform the exchanges and set the order
		if exchanges[0] != 0:
			for i in xrange(4):
				self.players[ind[i]].cards += exchanges[3-i]
				for card in exchanges[i]:
					self.players[ind[i]].cards.remove(card)
			for k in xrange(len(self.players)):
				self.order[self.players[k].out-1] = number_of_players - 1 - k
		for k in xrange(len(self.players)):
			self.players[k].out = 0
		self.history = history(number_of_players)
		time.sleep(0.5)
		print("New game starts.")
		return None

	def play_turn(self):
		
		"""
		Method that plays the turn for each player
		"""
					
        # This function is used to play a turn of each player
		actual_player = self.actual_player

		if self.passes == len(self.players) - 1 - self.counter:
			self.last = (0, 0)
			self.passes = 0


		# We check here if the player is still in the game
		if self.players[self.order[actual_player]].out == 0:

			# Here each agent player chooses an action according to different parameters
			move = self.players[self.order[actual_player]].choose(self.last,
                                                                  self.revolution,
                                                                  self.history,
                                                                  self.counter,
                                                                  (self.last[0] == rev_ranks[self.revolution][rank_max]))
            
			# Updating the heuristic
			self.heuristics[self.order[actual_player]] = move[0]
			
			# If the agent chooses to play, we update the stack and the history values
			if move[0] != 0:
				self.last = move
				self.history.update(self.order[actual_player], move)

            # Updating the hand
				self.players[self.order[actual_player]].play(move, self.revolution)
				self.passes = 0
				# Logging what has the player
				if self.verbose :
					print("Player "+str(self.order[actual_player])+" has thrown : " + str(move) + " and has " +
	                      str(len(self.players[self.order[actual_player]].cards)) + " cards left.")

					time.sleep(1)
				# Compute the reward
				reward = move[1] * rewards[self.revolution][move[0]]

				# Here we check if the agent/player has and empty hand
				if len(self.players[self.order[actual_player]].cards) == 0:
					if self.verbose :
						print("Player "+str(self.order[actual_player])+" is out.")

					# Lets leave this game
					self.counter += 1
                     # Increasing the counter to see how many people have left the game
					self.players[self.order[actual_player]].out = self.counter
					self.order[actual_player] = len(self.players) - self.players[self.order[actual_player]].out
					# Updating the final reward
					reward += self.final[self.counter]
					
					# Now we update all the status according to the counter
					if self.counter == 1:
						self.players[self.order[actual_player]].status = 'President'
					elif self.counter == 2:
						self.players[self.order[actual_player]].status = 'Vice-president'
					elif self.counter == len(self.players) - 1:
						self.players[self.order[actual_player]].status = 'Vice-trou'
					elif self.counter == len(self.players):
						self.players[self.order[actual_player]].status = 'Trou'
					else:
						self.players[self.order[actual_player]].status = 'People'

				# Let's use some reinforcement learning and learn!
				self.players[self.order[actual_player]].update(reward, self.last, self.history, self.revolution, self.counter)

            # This is the case where the Agent/player doesnt want to throw a card
			else:
				if self.verbose :
					print("Player "+str(self.order[actual_player]) + " passes his turn and has "
	                      + str(len(self.players[self.order[actual_player]].cards)) + " cards left.")
				self.passes += 1
        # Update who the player is
		if actual_player + 1 == len(self.players):
			self.actual_player = 0
		else:
			self.actual_player += 1
			
		return None
 

	def play_game(self):
					
		"""
		Method that plays the whole game.
		"""

		self.reset()
		while self.counter < len(self.players):
			self.play_turn()
												
												
		if self.verbose :
			for k in xrange(len(self.players)):
				print("Player " + str(k) + " has ended as the "+self.players[k].status+".")
		return None