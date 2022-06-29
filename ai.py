import numpy as np

class AI:
  def compute(self, board):
    x = np.random.randint(7)
    while not board.placeable(x):
      x = np.random.randint(7)
    return x
