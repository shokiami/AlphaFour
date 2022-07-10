import numpy as np

class Board:
  def __init__(self):
    self.mat = np.zeros((6, 7))
    self.count = 0
    self.player = 1
    self.history = []
  
  def placeable(self, x):
    return self.mat[0][x] == 0

  def place(self, x):
    if not self.placeable(x):
      raise Exception("Trying to place in already filled column.")
    y = 0
    while y < 6 and self.mat[y, x] == 0:
      y += 1
    y -= 1
    self.mat[y, x] = self.player
    self.count += 1
    self.player = 3 - self.player
    self.history.append((x, y))
    if self.four_in_a_row(x, y):
      return self.mat[y, x]
    elif self.count == 42:
      return 3
    else:
      return 0

  def undo(self):
    if len(self.history) == 0:
      raise Exception("No moves to undo.")
    x, y = self.history.pop()
    self.mat[y, x] = 0
    self.count -= 1
    self.player = 3 - self.player

  def four_in_a_row(self, x, y):
    for dx, dy in [(1, 0), (1, 1), (0, 1), (1, -1)]:
      in_a_row = 1
      for sgn in (-1, 1):
        x_ = x + sgn * dx
        y_ = y + sgn * dy
        while x_ > -1 and x_ < 7 and y_ > -1 and y_ < 6:
          if self.mat[y_, x_] != self.mat[y, x]:
            break
          in_a_row += 1
          x_ += sgn * dx
          y_ += sgn * dy
      if in_a_row >= 4:
        return True
    return False
