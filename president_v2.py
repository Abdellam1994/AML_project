import numpy as np
import time

from constantes import *
from utils import find_best, find_worst, game_reward


class history:
    
    def __init__(self, n):
        self.players = [{card: 0 for card in ranks[0].keys()} for k in xrange(n)]
        self.left = {card: 4 for card in ranks[0].keys()}
        self.left_cards = dict({pl: 52//n + 1 for pl in xrange(n) if(pl < 52 % n)}, **{pl: 52//n for pl in xrange(n) if(pl >= 52 % n)})
     
    def own(self, hand):
        for card in hand:
            self.left[card[0]] -= card[1]
        return
    
    def update(self, pl, move):
        if move[1] != 0:
            # Updating the cards played by the player
            self.players[pl][move[0]] += move[1]
            # Updating the number of cards of the value played
            self.left[move[0]] -= move[1]
            # Updating the cards left for the player
            self.left_cards[pl] -= move[1]
        return

# Player class with hand, status and an agent to make decisions
class player:
    
    def __init__(self, Agent, cards):
        self.agent = Agent
        self.cards = cards
        # np.array([0 for k in xrange(15)]+hand)
        self.status = 'People'
        self.out = 0
    
    # Method for determining the possible moves : Validated
    def possible_moves(self, last, revo=0, passe=False):
        if str(self.agent).split('.')[1].split(' ')[0] == "playerAgent":
           print(self.cards)
        pm = [(0, 0)]
        if passe:
            return pm
        # If the player doesn't initiate the turn
        if last[1] != 0:
            # Get the value of the last card(s)
            v = ranks[revo][last[0]]

            # Get the value of the cards in hand
            w = [ranks[revo][card] for card in self.cards]

            # Get the playable cards with their cardinality
            L = {n: np.sum([n == value for value in w]) for n in xrange(1 + v, 1 + rank_max)}

            # If the player is trou or vice-trou, the equality right activates
            if self.status in statuses[3:]:
                L[v] = np.sum([v == value for value in w])

            # Adding the possible moves : play the same number of cards as the last play
            for n in L.keys():
                if L[n] >= last[1]:
                    pm.append((rev_ranks[revo][n], last[1]))
        # If the player initiates the turn
        else:
            # Get the value of the cards in hand
            w = [ranks[revo][card] for card in self.cards]
            # Get the cardinality of the cards at hand
            L = {n : np.sum([n == value for value in w]) for n in set(w)}
            # Adding the possibe moves : play any possible number of any card at hand
            for n in L.keys():
                for k in xrange(L[n]):
                    pm.append((rev_ranks[revo][n], 1+k))
        return pm
    
    # Method for playing: remove the cards played : Validated
    def play(self, move, revo):
        if move == (0, 0):
            return
        else:
            for k in xrange(move[1]):
                self.cards.remove(move[0])
            return 
            
    # Method for choosing the action: relies on the agent used by the player
    # hand,history,order,pj,revo
    def choose(self, last, revo, history, counter, passe):
        # Last,cards,history,revo,counter
        self.agent.updateState(last, self.cards, history, revo, counter)
        return self.agent.choose(self.possible_moves(last, revo, passe))

    # Method to update the agent
    def update(self, reward, last, history, revo, counter):
        self.agent.update(reward, last, self.cards, history, revo, self.possible_moves(last, revo), counter)
        return
        
# Game class containing players each with their own agent they use to make decisions
class game:
    
    def __init__(self, n_player, agents, final):
        # The stack is empty
        self.last = (0, 0)
        
        # Setting the rewards
        self.final = game_reward(n_player, final)
        
        # Shuffling the cards
        np.random.shuffle(deck)
        
        # Counting the cards
        q = 52 // n_player
        r = 52 % n_player
        self.players = []

        # Creating the players and distributing the cards
        if r == 0:
            self.players += [player(agents[k], list(deck[k * q: (k+1) * q])) for k in xrange(n_player)]
        else:
            self.players += [player(agents[k], list(deck[k * (q+1): (k+1) * (q+1)])) for k in xrange(r)]
            self.players += [player(agents[k], list(deck[r * (q+1) + k * q: r * (q+1) + (k+1) * q])) for k in xrange(n_player-r)]
        
        # Setting the order at which the players play
        self.order = range(n_player)
        
        # No revolution at the begining of the game
        self.revo = 0
        
        # The first player starts
        self.actual_player = 0
        
        # History of cards played
        self.history = history(n_player)
        
        # Counter of people that left the game
        self.counter = 0
        
        # Counting the passes for the initiative transfer
        self.passes = 0

    # Validated
    def reset(self):
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
        self.revo = 0
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
                exchanges[2] = worst_cards(self.players[i].cards, 1)
                print("Player "+str(i)+" is the vice-president.")
            if self.players[i].status == 'President':
                ind[3] = i
                exchanges[3] = worst_cards(self.players[i].cards, 2)
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
        return
        
    def play_turn(self):
        pl = self.actual_player
        
        if self.passes == len(self.players) - 1 - self.counter:
            self.last = (0, 0)
            self.passes = 0
        
        # If the player hasn't left the game yet
        if self.players[self.order[pl]].out == 0:
            
            # The player chooses a move
            move = self.players[self.order[pl]].choose(self.last,
                                                       self.revo,
                                                       self.history,
                                                       self.counter,
                                                       (self.last[0] == rev_ranks[self.revo][rank_max]))
            
            # Updating the stack and the history if the player actually plays
            if move[0] != 0:
                self.last = move
                self.history.update(self.order[pl], move)
            
            # Removing the cards played from the player's hand
                self.players[self.order[pl]].play(move, self.revo)
                self.passes = 0
                print("Player "+str(self.order[pl])+" has thrown : " + str(move) + " and has " +
                      str(len(self.players[self.order[pl]].cards)) + " cards left.")
                time.sleep(1)
                
                # Getting the reward
                reward = move[1]*rewards[self.revo][move[0]]
                
                # In case the player has emptied his hand
                if len(self.players[self.order[pl]].cards) == 0:
                    print("Player "+str(self.order[pl])+" is out.")
                    # He leaves the game, updates his status and gets the final reward
                    self.counter += 1
                    self.players[self.order[pl]].out = self.counter
                    self.order[pl] = len(self.players) - self.players[self.order[pl]].out
                    reward += self.final[self.counter]
    
                    if self.counter == 1:
                        self.players[self.order[pl]].status = 'President'
                    elif self.counter == 2:
                        self.players[self.order[pl]].status = 'Vice-president'
                    elif self.counter == len(self.players) - 1:
                        self.players[self.order[pl]].status = 'Vice-trou'
                    elif self.counter == len(self.players):
                        self.players[self.order[pl]].status = 'Trou'
                    else:
                        self.players[self.order[pl]].status = 'People'
                
                # Learning from the move
                self.players[self.order[pl]].update(reward, self.last, self.history, self.revo, self.counter)
                 
            # The player throws no card
            else:
                print("Player "+str(self.order[pl])+" passes his turn and has "+str(len(self.players[self.order[pl]].cards))+" cards left.")
                self.passes += 1
                
        if pl+1 == len(self.players):
            self.actual_player = 0
        else:
            self.actual_player += 1
        return
        
    def play_game(self):
        self.reset()
        while self.counter < len(self.players):
            self.play_turn()
        for k in xrange(len(self.players)):
            print("Player " + str(k) + " has ended as the "+self.players[k].status+".")
        return