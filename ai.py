import numpy as np

class AI:
  def compute(self, board):
    x = np.random.randint(7)
    while not board[0, x] == 0:
      x = np.random.randint(7)
    return x
