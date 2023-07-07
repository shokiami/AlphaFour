from ai import AI
import numpy as np

state = np.array([
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [-1, 0, 0, 0, 0, 0, 0],
  [-1, 0, 0, 0, 0, 0, 0],
  [-1, 0, 0, 0, 0, 0, 0],
])

ai = AI()
print(ai.mcts_search(state))
