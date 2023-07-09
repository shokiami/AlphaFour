from game import init_state, get_valid_actions, get_next_state, is_terminal
from ai import AI
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame
from pygame import gfxdraw

def get_events():
  events = {'click': False, 'quit': False, 'r': False, 'q': False}
  for event in pygame.event.get():
    events['click'] = event.type == pygame.MOUSEBUTTONDOWN
    events['quit'] = event.type == pygame.QUIT
    if event.type == pygame.KEYDOWN:
      events['r'] = event.key == pygame.K_r
      events['q'] = event.key == pygame.K_q
  return events

def render(state, canvas, arrow_x):
  canvas.fill((255, 255, 255))
  gfxdraw.box(canvas, (0, 100, 700, 700), (255, 235, 0))
  for i in range(6):
    for j in range(7):
      color = (255, 255, 255)
      if state[i, j] == 1:
        color = (255, 0, 0)
      if state[i, j] == -1:
        color = (0, 0, 255)
      gfxdraw.aacircle(canvas, 100 * j + 50, 100 * i + 150, 40, color)
      gfxdraw.filled_circle(canvas, 100 * j + 50, 100 * i + 150, 40, color)
  if arrow_x != None:
    gfxdraw.aatrigon(canvas, arrow_x - 30, 30, arrow_x + 30, 30, arrow_x, 70, (255, 0, 0))
    gfxdraw.filled_trigon(canvas, arrow_x - 29, 31, arrow_x + 29, 31, arrow_x, 69, (255, 0, 0))
  pygame.display.update()

def main():
  pygame.init()
  pygame.display.set_caption('AlphaFour')
  canvas = pygame.display.set_mode((700, 700))
  state = init_state()
  player = 1
  arrow_x = 350
  ai = AI()
  terminal = False
  running = True
  while running:
    events = get_events()
    if not terminal:
      if player == 1:
        action = int(pygame.mouse.get_pos()[0] / 100)
        arrow_x = int(0.2 * (100 * action + 50) + 0.8 * arrow_x)
        if events['click'] and get_valid_actions(state)[action]:
          state = get_next_state(state, player, action)
          terminal, win = is_terminal(state, action)
          if terminal:
            if win:
              print('Red wins!')
            else:
              print('Draw game!')
          else:
            player = -player
      else:
        action = ai.compute(state)
        state = get_next_state(state, player, action)
        terminal, win = is_terminal(state, action)
        if terminal:
          if win:
            print('Blue wins!')
          else:
            print('Draw game!')
        else:
          player = -player
    render(state, canvas, arrow_x if player == 1 and not terminal else None)
    if events['r']:
      state = init_state()
      player = 1
      arrow_x = 350
      terminal = False
    if events['quit'] or events['q']:
      running = False

if __name__ == '__main__':
  main()
