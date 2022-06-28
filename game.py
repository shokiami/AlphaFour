import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

class Game:
  def __init__(self):
    pygame.init()
    pygame.display.set_caption("Connect 4")
    self.canvas = pygame.display.set_mode((700, 700))
    self.running = True
    self.board = np.zeros((6, 7))
    self.move = 1
    self.winner = 0
    self.arrow_x = 350
  
  def four_in_a_row(self, x0, y0, dx, dy):
    in_a_row = 1
    x = x0 - dx
    y = y0 - dy
    while x > -1 and x < 7 and y > -1 and y < 6:
      if self.board[y, x] != self.board[y0, x0]:
        break
      in_a_row += 1
      x -= dx
      y -= dy
    x = x0 + dx
    y = x0 + dy
    while x > -1 and x < 7 and y > -1 and y < 6:
      if self.board[y, x] != self.board[y0, x0]:
        break
      in_a_row += 1
      x += dx
      y += dy
    return in_a_row >= 4
  
  def placeable(self, x):
    return self.board[0][x] == 0

  def place(self, x):
    y = 0
    while y < 6 and self.board[y, x] == 0:
      y += 1
    y -= 1
    player = 2 - self.move % 2
    self.board[y, x] = player
    if self.four_in_a_row(x, y, 1, 0) or self.four_in_a_row(x, y, 1, 1) or self.four_in_a_row(x, y, 0, 1) or self.four_in_a_row(x, y, 1, -1):
      self.winner = player
    self.move += 1

  def update(self):
    x = int(pygame.mouse.get_pos()[0] / 100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.running = False
        if event.type == pygame.MOUSEBUTTONDOWN and self.placeable(x):
          self.place(x)
    self.arrow_x += 0.2 * (100 * x + 50 - self.arrow_x)

  def render(self):
    self.canvas.fill("white")
    pygame.draw.rect(self.canvas, "yellow", (0, 100, 700, 700))
    for x in range(7):
      for y in range(6):
        color = "white"
        if self.board[y, x] == 1:
          color = "red"
        if self.board[y, x] == 2:
          color = "blue"
        pygame.draw.circle(self.canvas, color, (100 * x + 50, 100 * y + 150), 40)
    if self.move % 2 == 1:
      color = "red"
    else:
      color = "blue"
    pygame.draw.polygon(self.canvas, color, [(self.arrow_x - 30, 30), (self.arrow_x + 30, 30), (self.arrow_x, 70)])
    pygame.display.update()
