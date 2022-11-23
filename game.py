from ai import AI
from board import Board
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
from pygame import gfxdraw

WHITE = (255, 255, 255)
YELLOW = (255, 235, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class Game:
  def __init__(self):
    pygame.init()
    pygame.display.set_caption("Connect 4")
    self.canvas = pygame.display.set_mode((700, 700))
    self.running = True
    self.board = Board()
    self.winner = 0
    self.arrow_x = 350
    self.ai = AI()

  def update(self):
    events = {"click": False, "quit": False, "r": False, "q": False}
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
          events["quit"] = True
        if event.type == pygame.MOUSEBUTTONDOWN:
          events["click"] = True
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_r:
            events["r"] = True
          if event.key == pygame.K_q:
            events["q"] = True
    if self.winner == 0:
      if self.board.player == 1:
        x = int(pygame.mouse.get_pos()[0] / 100)
        self.arrow_x = int(0.2 * (100 * x + 50) + 0.8 * self.arrow_x)
        if events["click"] and self.board.placeable(x):
          self.winner = self.board.place(x)
      else:
        x, q_val = self.ai.compute(self.board)
        self.winner = self.board.place(x)
        print(q_val)
    else:
      if self.winner == 1:
        print("Red wins!")
      elif self.winner == 2:
        print("Blue wins!")
      elif self.winner == 3:
        print("Draw game!")
      self.winner = -1  # pause game
    if events["r"]:
      self.board = Board()
      self.winner = 0
      self.arrow_x = 350
    if events["quit"] or events["q"]:
      self.running = False

  def render(self):
    self.canvas.fill(WHITE)
    gfxdraw.box(self.canvas, (0, 100, 700, 700), YELLOW)
    for x in range(7):
      for y in range(6):
        color = WHITE
        if self.board.mat[y, x] == 1:
          color = RED
        if self.board.mat[y, x] == 2:
          color = BLUE
        gfxdraw.aacircle(self.canvas, 100 * x + 50, 100 * y + 150, 40, color)
        gfxdraw.filled_circle(self.canvas, 100 * x + 50, 100 * y + 150, 40, color)
    if self.winner == 0 and self.board.player == 1:
      gfxdraw.aatrigon(self.canvas, self.arrow_x - 30, 30, self.arrow_x + 30, 30, self.arrow_x, 70, RED)
      gfxdraw.filled_trigon(self.canvas, self.arrow_x - 29, 31, self.arrow_x + 29, 31, self.arrow_x, 69, RED)
    pygame.display.update()
