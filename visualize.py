from main import render
from game import ConnectFour
from train import AI
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame
import numpy as np
from PIL import Image

PATH = 'test.gif'

def to_image(state, canvas):
  render(state, canvas)
  return Image.fromarray(np.array(pygame.surfarray.pixels3d(canvas)).swapaxes(0, 1))

def main():
  actions = [0, 1, 2]
  pygame.init()
  pygame.display.set_caption('AlphaFour')
  cf = ConnectFour()
  ai = AI(cf)
  state = cf.init_state()
  canvas = pygame.display.set_mode((700, 700))
  images = []
  images.append(to_image(state, canvas))
  for action in actions:
    state = cf.next_state(state, 1, action)
    images.append(to_image(state, canvas))
    terminal, win = cf.is_terminal(state, action)
    if terminal:
      break
    action = ai.compute(state)
    state = cf.next_state(state, -1, action)
    images.append(to_image(state, canvas))
    terminal, win = cf.is_terminal(state, action)
    if terminal:
      break
  images.insert(0, images.pop())
  images[0].save(PATH, format='GIF', append_images=images, save_all=True, duration=800)

if __name__ == '__main__':
  main()
