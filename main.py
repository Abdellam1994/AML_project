# -*- coding: utf-8 -*-


# Importing the different objects

from president import GAME
from agents import minAgent, maxAgent, minMaxAgent
from learning_agents import NNQL_Agent, LSTD_Agent, UCB1_Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


######################################################################
################## NEURAL NETWORK Q-LEARNING #########################
######################################################################


# Launching the game and gathering the rewards

NN_agent = NNQL_Agent(4)

agents = [minAgent(), NN_agent, maxAgent(), minMaxAgent()]


game = GAME(agents, verbose = False)

number_play = 10000

for i in tqdm(range(number_play)):
	game.play_game()
	
	
rewards_NNQL = np.array(NN_agent.rewards)

rewards_plot_NNQL = np.log(np.cumsum(rewards_NNQL))

np.savetxt("rewards_NNQL.txt", rewards_NNQL)

plt.figure()
plt.title("Obtained rewards for Neural Network Q_learning")
plt.xlabel("Turns")
plt.ylabel("Reward")
plt.plot(rewards_plot_NNQL)
plt.savefig("rewNNQL.jpg")
plt.show()


######################################################################
####################### LSTD Q-LEARNING ##############################
######################################################################	

# Launching the game and gathering the rewards

LS_agent = LSTD_Agent(4)

agents = [minAgent(), LS_agent, maxAgent(), minMaxAgent()]


game = GAME(agents, verbose = False)

number_play = 10000

for i in tqdm(range(number_play)):
	game.play_game()
	
	
rewards_LSTD = np.array(NN_agent.rewards)

rewards_plot_LSTD = np.log(np.cumsum(rewards_LSTD))

np.savetxt("rewards_LSTD.txt", rewards_LSTD)

plt.figure()
plt.title("Obtained rewards for LSTD")
plt.xlabel("Turns")
plt.ylabel("Reward")
plt.plot(rewards_plot_LSTD)
plt.savefig("rewLSTD.jpg")
plt.show()


######################################################################
############################# UCB1 ###################################
######################################################################	

# Launching the game and gathering the rewards

UC_agent = UCB1_Agent(4)

agents = [minAgent(), UC_agent, maxAgent(), minMaxAgent()]


game = GAME(agents, verbose = False)

number_play = 10000

for i in tqdm(range(number_play)):
	game.play_game()
	
	
rewards_UCB1 = np.array(NN_agent.rewards)

rewards_plot_UCB1 = np.log(np.cumsum(rewards_UCB1))

np.savetxt("rewards_UCB1.txt", rewards_UCB1)

plt.figure()
plt.title("Obtained rewards for UCB1")
plt.xlabel("Turns")
plt.ylabel("Reward")
plt.plot(rewards_plot_UCB1)
plt.savefig("rewUCB1.jpg")
plt.show()


######################################################################
####################### Comparison ###################################
######################################################################	

n = min((len(rewards_plot_NNQL), len(rewards_plot_LSTD),len(rewards_plot_UCB1)))

rewards_plot_NNQL = rewards_plot_NNQL[:n]
rewards_plot_LSTD = rewards_plot_LSTD[:n]
rewards_plot_UCB1 = rewards_plot_UCB1[:n]

plt.figure()
plt.title("Comparison of the different approaches")
plt.xlabel("Turns")
plt.ylabel("Reward")
plt.plot(rewards_plot_UCB1)
plt.plot(rewards_plot_LSTD)
plt.plot(rewards_plot_NNQL)
plt.savefig("comparison.jpg")
plt.show()