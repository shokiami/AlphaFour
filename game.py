import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame import gfxdraw
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
    for sgn in (-1, 1):
      x = x0 + sgn * dx
      y = y0 + sgn * dy
      while x > -1 and x < 7 and y > -1 and y < 6:
        if self.board[y, x] != self.board[y0, x0]:
          break
        in_a_row += 1
        x += sgn * dx
        y += sgn * dy
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
    self.move += 1
    if self.four_in_a_row(x, y, 1, 0) or self.four_in_a_row(x, y, 1, 1) or self.four_in_a_row(x, y, 0, 1) or self.four_in_a_row(x, y, 1, -1):
      self.winner = player
    elif self.move == 42:
      self.winner = 3

  def compute(self):
    x = np.random.randint(7)
    while not self.placeable(x):
      x = np.random.randint(7)
    pygame.time.wait(1000)
    return x

  def update(self):
    events = {"click": False, "quit": False, "r": False}
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
          events["quit"] = True
        if event.type == pygame.MOUSEBUTTONDOWN:
          events["click"] = True
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_r:
            events["r"] = True
    if self.winner == 0:
      if self.move % 2 == 1:
        x = int(pygame.mouse.get_pos()[0] / 100)
        self.arrow_x = int(0.2 * (100 * x + 50) + 0.8 * self.arrow_x)
        if events["click"] and self.placeable(x):
          self.place(x)
      else:
        x = self.compute()
        self.place(x)
    else:
      if self.winner == 1:
        print("Red wins!")
      elif self.winner == 2:
        print("Blue wins!")
      elif self.winner == 3:
        print("Draw game!")
      self.winner = -1  # pause game
    if events["r"]:
      self.board = np.zeros((6, 7))
      self.move = 1
      self.winner = 0
      self.arrow_x = 350
    if events["quit"]:
      self.running = False

  def render(self):
    self.canvas.fill((255, 255, 255))
    gfxdraw.box(self.canvas, (0, 100, 700, 700), (255, 255, 0))
    for x in range(7):
      for y in range(6):
        color = (255, 255, 255)
        if self.board[y, x] == 1:
          color = (255, 0, 0)
        if self.board[y, x] == 2:
          color = (0, 0, 255)
        gfxdraw.aacircle(self.canvas, 100 * x + 50, 100 * y + 150, 40, color)
        gfxdraw.filled_circle(self.canvas, 100 * x + 50, 100 * y + 150, 40, color)
    if self.move % 2 == 1:
      color = (255, 0, 0)
      gfxdraw.aatrigon(self.canvas, self.arrow_x - 30, 30, self.arrow_x + 30, 30, self.arrow_x, 70, color)
      gfxdraw.filled_trigon(self.canvas, self.arrow_x - 29, 31, self.arrow_x + 29, 31, self.arrow_x, 69, color)
    pygame.display.update()
