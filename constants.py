import numpy as np

# Normal ranks of cards
normal_ranks = {'3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, '2': 13}

# Reverse ranks if there is a revolution
revo_ranks = {k : 14 - v for k, v in normal_ranks.iteritems()}

# List of the two type of ranks
ranks = [normal_ranks, revo_ranks]

# Reverse correspondance for cards with ranks
rev_ranks = [{dic[key]: key for key in dic.keys()} for dic in ranks]

# Rewards for the cards played
rewards = [
    {'0': 0, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1, 'J': 1, 'Q': 1, 'K': 1, 'A': 1, '2': 1},
    {'0': 0, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1, 'J': 1, 'Q': 1, 'K': 1, 'A': 1, '2': 1}]


# Maximum rank
rank_max = 13

# Building the deck
values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
deck = np.array(values * 4)

# Different possible statuses
statuses = ['People','President','Vice-president','Trou','Vice-trou']
