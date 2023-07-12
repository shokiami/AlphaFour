from main import render
from game import ConnectFour
from ai import AlphaFour
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import shutil
import pygame
from PIL import Image

EXAMPLES = 'examples'

def to_image(state, canvas):
  render(state, canvas)
  return Image.fromarray(np.array(pygame.surfarray.pixels3d(canvas)).swapaxes(0, 1))

def save_gif(gen, actions, canvas):
  game = ConnectFour()
  state = game.init_state()
  ai = AlphaFour(game, gen)
  images = []
  images.append(to_image(state, canvas))
  for action in actions:
    state = game.next_state(state, 1, action)
    images.append(to_image(state, canvas))
    terminal, win = game.is_terminal(state, action)
    if terminal:
      break
    action = ai.compute(state)
    state = game.next_state(state, -1, action)
    images.append(to_image(state, canvas))
    terminal, win = game.is_terminal(state, action)
    if terminal:
      break
  images.insert(0, images.pop())
  images[0].save(os.path.join(EXAMPLES, f'{gen}.gif'), format='GIF', append_images=images, save_all=True, duration=800, loop=0)

def main():
  pygame.init()
  canvas = pygame.display.set_mode((700, 700))
  if os.path.isdir(EXAMPLES):
    shutil.rmtree(EXAMPLES)
  os.makedirs(EXAMPLES)
  save_gif( 0, [3, 2, 1, 0], canvas)
  save_gif(10, [3, 1, 3, 2, 4, 2, 5], canvas)

if __name__ == '__main__':
  main()
