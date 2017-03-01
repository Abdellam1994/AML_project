# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:26:59 2017

@author: Samed
"""

import numpy as np
import operator as op
import copy as cp
import time

#Ranks for the cards
ranks = [{'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7,'10':8,'J':9,'Q':10,'K':11,'A':12,'2':13},
         {'3':13,'4':12,'5':11,'6':10,'7':9,'8':8,'9':7,'10':6,'J':5,'Q':4,'K':3,'A':2,'2':1}]
    
#Reverse correspondance for cards with ranks
rev_ranks = [{dic[key]:key for key in dic.keys()} for dic in ranks]
              
#Rewards for the cards played
rewards = [{'0':0,'3':1,'4':1,'5':1,'6':1,'7':1,'8':1,'9':1,'10':1,'J':1,'Q':1,'K':1,'A':1,'2':1},
         {'0':0,'3':1,'4':1,'5':1,'6':1,'7':1,'8':1,'9':1,'10':1,'J':1,'Q':1,'K':1,'A':1,'2':1}]

#A function that sets the final rewards according to the number of players
def final_rewards(n,fr = [1000,500,0,-500,-1000]):
    R = {1:fr[0],2:fr[1],n-1:fr[3],n:fr[4]}
    if n>= 4:
        for k in xrange(3,n-1):
            R[k] = fr[2]
    return R
    
#Function used to get the n best cards out of a hand
def best_cards(cards,n,revo = 0):
    bc = []
    L = [ranks[revo][card] for card in cards]
    for k in xrange(n):
        i = np.argmax(L)
        bc.append(rev_ranks[revo][L[i]])
        L[i] = 0
    return bc
    
#Function used to get the n worst cards out of a hand
def worst_cards(cards,n,revo = 0):
    wc = []
    L = [ranks[revo][card] for card in cards]
    for k in xrange(n):
        i = np.argmin(L)
        wc.append(rev_ranks[revo][L[i]])
        L[i] = 13
    return wc
    
#Function to go back and forth from the neural network's representation of the
#Q-values to the actual Q-values through an affine transformation
def to_NN(Q,fr = [1000,500,0,-500,-1000]):
    nn = Q*(np.max(fr)+13-np.min(fr))
    nn += np.min(fr)
    return nn
    

def comb(p, n):
    r = min(p, n-p)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom
    
#Function to get the probabilities of each player having a certain number of a card
#for every card
def compute_probabilities(hand,history,order,pj,revo):
    probas = []
    players_left = 0
    for pl in xrange(len(order)):
        if history.left[pl]!=0:
            players_left += 1
    if players_left>1:
        for k in order:
            if k!= pj:
                hist = history.players[k]
                for card in ranks[0].keys():
                    m = hand[card]
                    p = 4 - history.left[card]
                    cj = hist[card]
                    lim = min(4-m-p,cj)
                    for i in xrange(1+lim):
                        probas.append(comb(i,lim) * (1./players_left)**i  * (1-1./players_left)**(lim-i))
                    if lim<4:
                        for i in xrange(1+lim,5):
                            probas.append(0.)
    else:
        for k in order:
            if k!=pj:
                for card in ranks[0].keys():
                    for i in xrange(5):
                        probas.append(0.)
    probas.append(players_left)
    probas.append(revo)
    return np.array(probas)
                
class history:
    
    def __init__(self,n):
        self.players = [ {card : 0 for card in ranks[0].keys()} for k in xrange(n) ]
        self.left = {card : 4 for card in ranks[0].keys()}
        self.left_cards = dict({pl : 52//n + 1 for pl in xrange(n) if( pl < 52%n)},**{pl : 52//n for pl in xrange(n) if(pl>=52%n)})
     
    def own(self,hand):
        for card in hand:
            self.left[card[0]] -= card[1]
        return
    
    def update(self,pl,move):
        if move[1] != 0:
            #Updating the cards played by the player
            self.players[pl][move[0]] += move[1]
            #Updating the number of cards of the value played
            self.left[move[0]] -= move[1]
            #Updating the cards left for the player
            self.left_cards[pl] -= move[1]
        return
        
#Maximum rank
rank_max = 13

#Building the deck
values = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
deck = np.array(values * 4)

#Different possible statuses
statuses = ['People','President','Vice-president','Trou','Vice-trou']

#Neural network class to use for approximating the Q-values
class NN:
    
    def __init__(self,L):
        self.coefs = [np.random.normal(0.,1.,(L[i+1],L[i])) for i in xrange(len(L)-1)]
                      
    def train(self,x,t,lr):
        inter = [x]
        grad = [np.zeros(K.shape) for K in self.coefs]

        #Forward propagation
        for i in xrange(len(self.coefs)):
            v = self.coefs[i].dot(inter[-1]) 
            inter.append(1./(1+np.exp(-v)))
        
        #Backward propagation
        forward = inter[-1]
        delta = forward*(1-forward)*(t-forward)
        grad[-1] = delta * inter[-2]
        if len(self.coefs)>=2:
            for i in range(-2,-len(self.coefs)-1,-1):
                delta = inter[i]*(1-inter[i])*(np.transpose(self.coefs[i+1]).dot(delta))
                grad[i] = delta.reshape([delta.shape[0],1]).dot(inter[i-1].reshape(1,inter[i-1].shape[0]))
        
        #Updating the weights
        self.coefs = [self.coefs[i] - lr*grad[i] for i in xrange(len(self.coefs))]
        return inter[-1]
        
    def predict(self,x):
        inter = x
        
        #Forward propagation
        for i in xrange(len(self.coefs)):
            v = self.coefs[i].dot(inter)
            inter = 1./(1+np.exp(-v))
        
        return inter

        
class minAgent:
    #Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        return
      
    #There is also no need to update the state
    def updateState(self,last,cards,history,revo,counter):
        self.revo = revo
    
    #Here we choose the simple strategy of throwing the lowest value possible with the highest number possible
    def choose(self,pm):
        if len(pm)>1:
            values = np.array([[ranks[self.revo][move[0]],move[1]] for move in pm if move != (0,0)])
            card = np.min(values[:,0])
            inds = (values[:,0]==card)
            move = (rev_ranks[self.revo][card],np.max(values[inds,1]))
            return move
        else:
            return (0,0)
            
    #We update the revolution
    def update(self,reward,last,cards,history,revo,moves,counter):
        self.updateState(last,cards,history,revo,counter)
        return
        
class maxAgent:
    #Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        return
      
    #There is also no need to update the state
    def updateState(self,last,cards,history,revo,counter):
        self.revo = revo
    
    #Here we choose the simple strategy of throwing the lowest value possible with the highest number possible
    def choose(self,pm):
        if len(pm)>1:
            values = np.array([[ranks[self.revo][move[0]],move[1]] for move in pm if move != (0,0)])
            card = np.max(values[:,0])
            inds = (values[:,0]==card)
            move = (rev_ranks[self.revo][card],np.max(values[inds,1]))
            return move
        else:
            return (0,0)
            
class playerAgent:
  
    def __init__(self):
      self.last = (0,0)
      self.revo = 0
      return
      
    def updateState(self,last,cards,history,revo,counter):
        self.revo = revo
        self.last = last
        return
      
    def choose(self,pm):
        print("Last card thrown : "+str(self.last))
        if len(pm)!=1:
            pcn = {}
            for move in pm:
                if move[0]!=0:
                    if move[0] in pcn.keys():
                        pcn[move[0]].append(move[1])
                    else:
                        pcn[move[0]] = [move[1]]
                else:
                    pcn['0'] = [0]
            print(pcn)
            move = []
            while len(move)==0:
              card = str(raw_input("Choose a card\n"))
              if card in pcn.keys():
                  move.append(card)
            while len(move)==1:
                if self.last==(0,0):
                    number = input("Choose a number of cards\n")
                    if int(number) in pcn[move[0]]:
                        move.append(number)
                    else:
                        print("You can't throw this number of cards")
                else:
                    move.append(pcn[move[0]][0])
            return tuple(move)
        else:
            return (0,0)
        
            
    #We update the revolution
    def update(self,reward,last,cards,history,revo,moves,counter):
        self.updateState(last,cards,history,revo,counter)
        return
        
class minMaxAgent:
    
    #Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        self.minMax = 0
        return
      
    #There is also no need to update the state
    def updateState(self,last,cards,history,revo,counter):
        self.revo = revo
    
    #Here we choose the simple strategy of alternating between the lowest and largest values
    def choose(self,pm):
        if len(pm)>1:
            if self.minMax == 0:
                values = np.array([[ranks[self.revo][move[0]],move[1]] for move in pm if move!=(0,0)])
                card = np.min(values[:,0])
                inds = (values[:,0]==card)
                move = (rev_ranks[self.revo][card],np.max(values[inds,1]))
                return move
            else:
                values = np.array([[ranks[self.revo][move[0]],move[1]] for move in pm if move!=(0,0)])
                card = np.max(values[:,0])
                inds = (values[:,0]==card)
                move = (rev_ranks[self.revo][card],np.max(values[inds,1]))
                return move
        else:
            return (0,0)
            
    #We update the revolution
    def update(self,reward,last,cards,history,revo,moves,counter):
        self.minMax = 1 - self.minMax
        self.updateState(last,cards,history,revo,counter)
        return
        
#Player class with hand, status and an agent to make decisions
class player:
    
    def __init__(self,Agent,cards):
        self.agent = Agent
        self.cards = cards
        #np.array([0 for k in xrange(15)]+hand)
        self.status = 'People'
        self.out = 0
    
    #Method for determining the possible moves : Validated
    def possible_moves(self,last, revo = 0, passe = False):
        if str(self.agent).split('.')[1].split(' ')[0] == "playerAgent":
           print(self.cards)
        pm = [(0,0)]
        if passe == True:
            return pm
        #If the player doesn't initiate the turn
        if last[1]!=0:
            #Get the value of the last card(s)
            v = ranks[revo][last[0]]
            #Get the value of the cards in hand
            w = [ranks[revo][card] for card in self.cards]
            #Get the playable cards with their cardinality
            L = {n : np.sum([n==value for value in w]) for n in xrange(1+v,1+rank_max)}
            #If the player is trou or vice-trou, the equality right activates
            if self.status in statuses[3:]:
                L[v] = np.sum([v == value for value in w])
            #Adding the possible moves : play the same number of cards as the last play
            for n in L.keys():
                if L[n] >= last[1]:
                    pm.append((rev_ranks[revo][n],last[1]))
        #If the player initiates the turn
        else:
            #Get the value of the cards in hand
            w = [ranks[revo][card] for card in self.cards]
            #Get the cardinality of the cards at hand
            L = {n : np.sum([n==value for value in w]) for n in set(w)}
            #Adding the possibe moves : play any possible number of any card at hand
            for n in L.keys():
                for k in xrange(L[n]):
                    pm.append((rev_ranks[revo][n],1+k))
        return pm
    
    #Method for playing: remove the cards played : Validated
    def play(self,move,revo):
        if move == (0,0):
            return
        else:
            for k in xrange(move[1]):
                self.cards.remove(move[0])
            return 
            
    #Method for choosing the action: relies on the agent used by the player
    #hand,history,order,pj,revo
    def choose(self,last,revo,history,counter,passe):
        #last,cards,history,revo,counter
        self.agent.updateState(last,self.cards,history,revo,counter)
        return self.agent.choose(self.possible_moves(last,revo,passe))

    #Method to update the agent
    def update(self,reward,last,history,revo,counter):
        self.agent.update(reward,last,self.cards,history,revo,self.possible_moves(last,revo),counter)
        return
        
#Game class containing players each with their own agent they use to make decisions
class game:
    
    def __init__(self,n_player,agents,final):
        #The stack is empty
        self.last = (0,0)
        
        #Setting the rewards
        self.final = final_rewards(n_player,final)
        
        #Shuffling the cards
        np.random.shuffle(deck)
        
        #Counting the cards
        q = 52//n_player
        r = 52%n_player
        self.players = []

        #Creating the players and distributing the cards
        if r==0:
            self.players += [player(agents[k],list(deck[k*q:(k+1)*q])) for k in xrange(n_player)]
        else:
            self.players += [player(agents[k],list(deck[k*(q+1):(k+1)*(q+1)])) for k in xrange(r)]
            self.players += [player(agents[k],list(deck[r*(q+1)+k*q:r*(q+1)+(k+1)*q])) for k in xrange(n_player-r)]
        
        #Setting the order at which the players play
        self.order = range(n_player)
        
        #No revolution at the begining of the game
        self.revo = 0
        
        #The first player starts
        self.actual_player = 0
        
        #History of cards played
        self.history = history(n_player)
        
        #Counter of people that left the game
        self.counter = 0
        
        #Counting the passes for the initiative transfer
        self.passes = 0
        
    def reset(self):#Validated
        #Resetting the game
        self.last = (0,0)
        np.random.shuffle(deck)
        q = 52//len(self.players)
        r = 52%len(self.players)
        if r!=0:
            for k in xrange(r):
                self.players[k].cards = list(deck[(q+1)*k:(q+1)*(k+1)])
            for k in xrange(len(self.players)-r):
                self.players[r+k].cards = list(deck[(q+1)*r+q*k:(q+1)*r+q*(k+1)])
        else:
            for k in xrange(len(self.players)):
                self.players[k].cards = list(deck[q*k:q*(k+1)])
        self.revo = 0
        self.counter = 0
        #Doing the exchange of cards for the two highest and two lowest ranked
        ind = [0 for k in xrange(4)]
        exchanges = [0 for k in xrange(4)]
        #Choosing the cards to be exchanged
        for i in xrange(len(self.players)):
            if self.players[i].status == 'Trou':
                ind[0] = i
                exchanges[0] = best_cards(self.players[i].cards,2)
                print("Player "+str(i)+" is the trou.")
            if self.players[i].status == 'Vice-trou':
                ind[1]= i
                exchanges[1] = best_cards(self.players[i].cards,1)
                print("Player "+str(i)+" is the vice-trou.")
            if self.players[i].status == 'Vice-president':
                ind[2] = i
                exchanges[2] = worst_cards(self.players[i].cards,1)
                print("Player "+str(i)+" is the vice-president.")
            if self.players[i].status == 'President':
                ind[3] = i
                exchanges[3] = worst_cards(self.players[i].cards,2)
                print("Player "+str(i)+" is the president.")
        #If we're not at the begining of the game, we perform the exchanges and set the order
        if exchanges[0]!=0:
            for i in xrange(4):
                self.players[ind[i]].cards += exchanges[3-i]
                for card in exchanges[i]:
                    self.players[ind[i]].cards.remove(card)
            for k in xrange(len(self.players)):
                self.order[self.players[k].out-1] = len(self.players) - 1 - k
        for k in xrange(len(self.players)):
            self.players[k].out = 0
        self.history = history(len(self.players))
        time.sleep(0.5)
        print("New game starts.")
        return
        
    def play_turn(self):
        pl = self.actual_player
        
        if self.passes == len(self.players) - 1 - self.counter:
            self.last = (0,0)
            self.passes = 0
        
        #If the player hasn't left the game yet
        if self.players[self.order[pl]].out == 0:
            
            #The player chooses a move
            move = self.players[self.order[pl]].choose(self.last,self.revo,self.history,self.counter,(self.last[0]==rev_ranks[self.revo][rank_max]))
            
            #Updating the stack and the history if the player actually plays
            if move[0]!=0:
                self.last = move
                self.history.update(self.order[pl],move)
            
            #Removing the cards played from the player's hand
                self.players[self.order[pl]].play(move,self.revo)
                self.passes = 0
                print("Player "+str(self.order[pl])+" has thrown : "+str(move)+" and has "+str(len(self.players[self.order[pl]].cards))+" cards left.")
                time.sleep(1)
                
                #Getting the reward
                reward = move[1]*rewards[self.revo][move[0]]
                
                #In case the player has emptied his hand
                if len(self.players[self.order[pl]].cards)==0:
                    print("Player "+str(self.order[pl])+" is out.")
                    #He leaves the game, updates his status and gets the final reward
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
                
                #Learning from the move
                self.players[self.order[pl]].update(reward,self.last,self.history,self.revo,self.counter)
                 
            #The player throws no card
            else:
                print("Player "+str(self.order[pl])+" passes his turn and has "+str(len(self.players[self.order[pl]].cards))+" cards left.")
                self.passes += 1
                
        if pl+1==len(self.players):
            self.actual_player = 0
        else:
            self.actual_player += 1
        return
        
    def play_game(self):
        self.reset()
        while self.counter<len(self.players):
            self.play_turn()
        for k in xrange(len(self.players)):
            print("Player " +str(k)+" has ended as the "+self.players[k].status+".")
        return