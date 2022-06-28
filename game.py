import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

class Game:
  def __init__(self):
    pygame.init()
    pygame.display.set_caption("Connect 4")
    self.canvas = pygame.display.set_mode((800, 600))
    self.running = True
    self.board = np.zeros((6, 7))
    self.move = 1
    self.winner = 0
  
  def four_in_a_row(self, x0, y0, dx, dy):
    x = x0 - dx
    y = y0 - dy
    while x > -1 and x < 7 and y > -1 and y < 6 and self.board[x, y] == self.board[x0, y0]:
      x -= dx
      y -= dy
    in_a_row = 0
    x += dx
    y += dy
    while x > -1 and x < 7 and y > -1 and y < 6 and self.board[x, y] == self.board[x0, y0]:
      x += dx
      y += dy
      in_a_row += 1
    return in_a_row >= 4
  
  def placeable(self, x):
    return self.board[0][x] == 0

  def place(self, x):
    y = -1
    while y < 6 and self.board[y + 1, x] == 0:
      y += 1
    if y != -1:
      player = self.move % 2
      self.board[y, x] = player
      if self.four_in_a_row(x, y, 1, 0) or self.four_in_a_row(x, y, 1, 1) or self.four_in_a_row(x, y, 0, 1) or self.four_in_a_row(x, y, 1, -1):
        self.winner = player
      self.move += 1

  def update(self):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.running = False

  def render(self):
    self.canvas.fill("white")
    pygame.draw.rect(self.canvas, "red", pygame.Rect(0, 0, 100, 100))
    pygame.display.update()
