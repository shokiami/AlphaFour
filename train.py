from ai import AI
import numpy as np

state = np.array([
  [ 0,  0,  0,  0,  0,  0,  0],
  [ 0,  0,  0,  0,  0,  0,  0],
  [ 0,  0,  0,  0,  0,  0,  0],
  [ 0,  -1,  1,  0,  0,  0,  0],
  [ 1, -1,  1,  0,  0,  0,  0],
  [-1, -1, -1,  1,  1,  0,  0],
])

ai = AI()
ai.learn()
print(ai.compute(state))
