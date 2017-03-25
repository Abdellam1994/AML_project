# -*- coding: utf-8 -*-
import sys

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from president import GAME
from agents import minAgent, maxAgent, minMaxAgent
from learning_agents import NNQL_Agent, LSTD_Agent, UCB1_Agent

# Choosing the agent between lstd, ucb1, nnql
if __name__ == '__main__':

    number_play = sys.argv[1]
    agent = sys.argv[2]
    compare = sys.argv[3]

    if agent == 'ucb1':
        OurAgent = UCB1_Agent(4)
    elif agent == 'lstd':
        OurAgent = LSTD_Agent(4)
    else:
        OurAgent = NNQL_Agent(4)

    # Launching the game and gathering the rewards

    agents = [minAgent(), OurAgent, maxAgent(), minMaxAgent()]

    game = GAME(agents, verbose=False)

    number_play = 1000

    for i in tqdm(range(number_play)):
        game.play_game()

    rewards_ = np.array(OurAgent.rewards)

    rewards_plot_ = np.log(np.cumsum(rewards_))

    np.savetxt("rewards_{}.txt".format(agent), rewards_)

    if agent == 'ucb1':
        title = "Obtained rewards for UCB1"
    elif agent == 'lstd':
        title = "Obtained rewards for LSTD"
    else:
        title = "Obtained rewards for Neural Network Q_learning"

    plt.figure()
    plt.title(title)
    plt.xlabel("Turns")
    plt.ylabel("Reward")
    plt.plot(rewards_plot_)
    plt.savefig("rew_{}.jpg".format(agent))
    plt.show()

    df_rewards_plot_ = pd.DataFrame(rewards_plot_)
    df_rewards_plot_.to_csv('rew_{}.csv'.format(agent))
    ######################################################################
    ####################### Comparison ###################################
    ######################################################################

    if compare == 'compare':

        rewards_plot_NNQL = np.array(pd.read_csv('rew_nnql.csv'))
        rewards_plot_LSTD = np.array(pd.read_csv('rew_ucb1.csv'))
        rewards_plot_UCB1 = np.array(pd.read_csv('rew_lstd.csv'))

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