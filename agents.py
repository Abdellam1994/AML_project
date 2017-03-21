import operator as op

from constants import *





	

class minAgent:
    # Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        return

    # There is also no need to update the state
    def updateState(self, last, cards, history, revo, counter):
        self.revo = revo

    # Here we choose the simple strategy of throwing the lowest value possible with the highest number possible
    def choose(self, pm):
        if len(pm) > 1:
            values = np.array([[ranks[self.revo][move[0]], move[1]] for move in pm if move != (0, 0)])
            card = np.min(values[:, 0])
            inds = (values[:, 0] == card)
            move = (rev_ranks[self.revo][card], np.max(values[inds, 1]))
            return move
        else:
            return (0, 0)

    # We update the revolution
    def update(self, reward, last, cards, history, revo, moves, counter):
        self.updateState(last, cards, history, revo, counter)
        return


class maxAgent:
    # Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        return

    # There is also no need to update the state
    def updateState(self, last, cards, history, revo, counter):
        self.revo = revo

    # Here we choose the simple strategy of throwing the lowest value possible with the highest number possible
    def choose(self, pm):
        if len(pm) > 1:
            values = np.array([[ranks[self.revo][move[0]], move[1]] for move in pm if move != (0, 0)])
            card = np.max(values[:, 0])
            inds = (values[:, 0] == card)
            move = (rev_ranks[self.revo][card], np.max(values[inds, 1]))
            return move
        else:
            return (0, 0)


class RealPlayer:
    def __init__(self):
        self.last = (0, 0)
        self.revo = 0
        return

    def updateState(self, last, cards, history, revo, counter):
        self.revo = revo
        self.last = last
        return

    def choose(self, pm):
        print("Last card thrown : " + str(self.last))
        if len(pm) != 1:
            pcn = {}
            for move in pm:
                if move[0] != 0:
                    if move[0] in pcn.keys():
                        pcn[move[0]].append(move[1])
                    else:
                        pcn[move[0]] = [move[1]]
                else:
                    pcn['0'] = [0]
            print(pcn)
            move = []
            while len(move) == 0:
                card = str(raw_input("Choose a card\n"))
                if card in pcn.keys():
                    move.append(card)
            while len(move) == 1:
                if self.last == (0, 0):
                    number = input("Choose a number of cards\n")
                    if int(number) in pcn[move[0]]:
                        move.append(number)
                    else:
                        print("You can't throw this number of cards")
                else:
                    move.append(pcn[move[0]][0])
            return tuple(move)
        else:
            return (0, 0)

    # We update the revolution
    def update(self, reward, last, cards, history, revo, moves, counter):
        self.updateState(last, cards, history, revo, counter)
        return


class minMaxAgent:
    # Here there is no need for parameters as this agent is a deterministic one
    def __init__(self):
        self.revo = 0
        self.minMax = 0
        return

    # There is also no need to update the state
    def updateState(self, last, cards, history, revo, counter):
        self.revo = revo

    # Here we choose the simple strategy of alternating between the lowest and largest values
    def choose(self, pm):
        if len(pm) > 1:
            if self.minMax == 0:
                values = np.array([[ranks[self.revo][move[0]], move[1]] for move in pm if move != (0, 0)])
                card = np.min(values[:, 0])
                inds = (values[:, 0] == card)
                move = (rev_ranks[self.revo][card], np.max(values[inds, 1]))
                return move
            else:
                values = np.array([[ranks[self.revo][move[0]], move[1]] for move in pm if move != (0, 0)])
                card = np.max(values[:, 0])
                inds = (values[:, 0] == card)
                move = (rev_ranks[self.revo][card], np.max(values[inds, 1]))
                return move
        else:
            return (0, 0)

    # We update the revolution
    def update(self, reward, last, cards, history, revo, moves, counter):
        self.minMax = 1 - self.minMax
        self.updateState(last, cards, history, revo, counter)
        return