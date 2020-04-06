import numpy as np
import pickle
import time

BOARD_ROWS = 3
BOARD_COLS = 3

class Board:
    def __init__(self, boardState = None):
        self.state = np.zeros((BOARD_ROWS, BOARD_COLS)) if boardState is None else boardState
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.state.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.state[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.state[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(self.state[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.state[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.state[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.state[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.state[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.state[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # board reset
    def reset(self):
        self.state = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1


    def showBoard(self):
        # p1: x  p2: o
        print('  ' + '  0   1   2  ')
        for i in range(0, BOARD_ROWS):
            print('  ' + '-------------')
            out = str(i) + ' ' + '| '
            for j in range(0, BOARD_COLS):
                if self.state[i, j] == 1:
                    token = 'x'
                if self.state[i, j] == -1:
                    token = 'o'
                if self.state[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('  ' + '-------------')


class Judge:
    def __init__(self, p1, p2):
        self.board = Board()
        self.p1 = p1
        self.p2 = p2

    def train(self, rounds=100):
        for i in range(1, rounds+1):
            self.board.reset() # start from empty board
            while not self.board.isEnd:
                # Player 1's turn
                positions = self.board.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board)
                # take action and upate board state
                self.board.updateState(p1_action)
                # check board status if it is end
                win = self.board.winner()

                if not self.board.isEnd:
                    # Player 1 did not win or draw
                    # -> Player 2's turn
                    positions = self.board.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board)
                    self.board.updateState(p2_action)
                    # check board status if it is end
                    win = self.board.winner()
            if i % 1000 == 0:
                print("{} partite giocate...".format(i))


    # play with human
    def play(self):
        while not self.board.isEnd:
            # Player 1
            positions = self.board.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board)
            # take action and upate board state
            self.board.updateState(p1_action)
            time.sleep(2)
            self.board.showBoard()
            # check board status if it is end
            win = self.board.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "ha vinto!")
                else:
                    print("Patta!")
                self.board.reset()
                break

            else:
                # Player 2
                positions = self.board.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.board.updateState(p2_action)
                self.board.showBoard()
                win = self.board.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "ha vinto!")
                    else:
                        print("Patta!")
                    self.board.reset()
                    break


class Player:
    def __init__(self, name, exploration_rate=0.01):
        self.name = name
        self.learning_rate = 0.5
        self.exploration_rate = exploration_rate
        self.states_value = {}  # state -> value


    def chooseAction(self, positions, current_board):
        current_state = current_board.state
        symbol = current_board.playerSymbol
        if np.random.uniform(0, 1) <= self.exploration_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_state = current_state.copy()
                next_state[p] = symbol

                next_board = Board(next_state)
                next_boardHash = next_board.getHash()

                if self.states_value.get(next_boardHash) is None:
                    # if yet unexplored state
                    if next_board.winner() == symbol: # winning board states
                        # record as winning state
                        self.states_value[next_boardHash] = 1
                    elif next_board.winner() == (-1)*symbol: # loosing board states
                        # record as losing state
                        self.states_value[next_boardHash] = 0
                    else: # same odds to win or loose
                        # record as neutral state
                        self.states_value[next_boardHash] = 0.5

                value = self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p

            # update current state value
            if self.states_value.get(current_board.getHash()) is None:
                # cannot be either a win or a loss -> set current value to 0.5
                self.states_value[current_board.getHash()] = 0.5
            self.states_value[current_board.getHash()] += self.learning_rate * (self.states_value[next_boardHash] - self.states_value[current_board.getHash()])

        # print("{} takes action {}".format(self.name, action))
        return action


    def savePolicy(self):
        fw = open('policy_' + str(self.name) + '.pol', 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Inserisci la riga della tua mossa [0-2]: "))
            col = int(input("Inserisci la colonna della tua mossa [0-2]: "))
            action = (row, col)
            if action in positions:
                return action

if __name__ == "__main__":
    # training
    p1 = Player("p1")
    p2 = Player("p2")

    jdg = Judge(p1, p2)
    print("Allenamento:")
    jdg.train(20000)
    print("...terminato.")
    p1.savePolicy()

    # play with human
    p1 = Player("Computer", exploration_rate=0)
    p1.loadPolicy("policy_p1.pol")

    p2 = HumanPlayer("Uomo")

    jdg = Judge(p1, p2)
    jdg.play()
