import numpy as np

def init_state():
  return np.zeros((6, 7), dtype=np.int8)

def get_valid_actions(state):
  return state[0] == 0

def get_next_state(state, player, action):
  next_state = np.copy(state)
  row = np.max(np.where(next_state[:, action] == 0))
  next_state[row, action] = player
  return next_state

def is_terminal(state, prev_action):
  if prev_action == None:
    return False, False
  j0 = prev_action
  i0 = np.min(np.where(state[:, j0] != 0))
  player = state[i0, j0]
  for di, dj in [(1, 0), (1, 1), (0, 1), (1, -1)]:
    n = 1
    for sgn in (-1, 1):
      i = i0 + sgn * di
      j = j0 + sgn * dj
      while i >= 0 and i < 6 and j >= 0 and j < 7:
        if state[i, j] != player:
          break
        n += 1
        i += sgn * di
        j += sgn * dj
    if n >= 4:
      return True, True
  return np.sum(get_valid_actions(state)) == 0, False
