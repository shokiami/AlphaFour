from board import Board
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NUM_GAMES = 10000
LEARNING_RATE = 0.0001
EPSILON = 0.1
GAMMA = 0.9

class QNet(nn.Module):
  def __init__(self):
    super(QNet, self).__init__()
    self.conv = nn.Conv2d(1, 32, 4, 1, 1)
    self.fc = nn.Linear(960, 7)

  def forward(self, action):
    action = self.conv(action)
    action = torch.relu(action)
    action = torch.flatten(action)
    action = self.fc(action)
    return action

class AI:
  def __init__(self):
    self.qnet = QNet()

  def board_to_state(self, board):
    mat = board.mat.copy()
    mat[mat == 2] = -1.0
    if board.player == 2:
      mat *= -1.0
    return torch.FloatTensor(mat).unsqueeze(0)

  def compute_move(self, board):
    state = self.board_to_state(board)
    with torch.no_grad():
      qvals = self.qnet(state).numpy()
    action = self.safe_argmax(qvals, board)
    qval = qvals[action]
    return action, qval

  def safe_argmax(self, qvals, board):
    for action in np.argsort(qvals):
      if board.placeable(action):
        return action

  def random_move(self, board):
    action = np.random.randint(7)
    while not board.placeable(action):
      action = np.random.randint(7)
    return action

  def update(self, state, action, reward, optimizer):
    qvals = self.qnet(state)
    new_qvals = qvals.detach().clone()
    new_qvals[action] = reward
    optimizer.zero_grad()
    loss = F.mse_loss(qvals, new_qvals)
    loss.backward()
    optimizer.step()
    return loss.item()

  def train(self):
    self.qnet.train()
    optimizer = optim.Adam(self.qnet.parameters(), LEARNING_RATE)

    for i in range(NUM_GAMES):
      board = Board()
      losses = []

      state_p = None
      action_p = None
      state_pp = None
      action_pp = None

      while board.winner == 0:
        state = self.board_to_state(board)
        with torch.no_grad():
          qvals = self.qnet(state).numpy()

        action = self.safe_argmax(qvals, board)

        if action_pp:
          reward = 1.0 + GAMMA * qvals[action]
          losses.append(self.update(state_pp, action_pp, reward, optimizer))

        if np.random.rand() < EPSILON:
          action = self.random_move(board)

        board.place(action)

        state_pp = state_p
        action_pp = action_p
        state_p = state
        action_p = action

      if board.winner == 3:
        losses.append(self.update(state_pp, action_pp, 0.0, optimizer))
        losses.append(self.update(state_p, action_p, 0.0, optimizer))
      else:
        losses.append(self.update(state_pp, action_pp, -100.0, optimizer))
        losses.append(self.update(state_p, action_p, 100.0, optimizer))

      print(f'game: {i + 1}/{NUM_GAMES}, winner: {int(board.winner)}, loss: {np.mean(losses)}')

    print('done training!')
    self.qnet.eval()
