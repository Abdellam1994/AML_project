# AML_project

Our project is about a card game named President which is originally a
 Japanese card game. The rules of the game could be found here :
https://en.wikipedia.org/wiki/President_(card_game)
We have chosen to focus on this difficult challenge because it is a game
 that we are used to play but we have never found a winning strategy
  that outperformed the others. 
  
  Therefore, it appears as interesting and challenging to see how
   reinforcement learning algorithms perform in this case.

We coded our own environement.

To launch the learning with either the Neural Network agent, or 
the UCB1 or the LSTD Agent, just open ipython and run the following:

`run main.py 1000 lstd compare`

`1000` is used to choose the number of plays

To choose the agent, it has to be either `lstd`, `ucb1` or `nnql`.

The `compare` argument can be either `compare` or `nocompare`.
 It is can be used if and only if you already did the 
learning with the 3 agents before, and compares the 3 cumulative rewards