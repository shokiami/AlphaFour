from board import Board
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import csv
import matplotlib.pyplot as plt

LEARNING_RATE = 0.0001
EPSILON = 0.1
GAMMA = 0.9
WIN_REWARD = 10.0
LIVING_REWARD = 0.1

MODEL = 'qnet.pt'
LOSSES_CSV = 'losses.csv'
LOSSES_PLOT = 'losses.png'

class QNet(nn.Module):
  def __init__(self):
    super(QNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, 4, 1, 1)
    self.conv2 = nn.Conv2d(16, 32, 4, 1, 1)
    self.conv3 = nn.Conv2d(32, 64, 4, 1, 1)
    self.fc = nn.Linear(768, 7)

  def forward(self, x):
    x = self.conv1(x)
    x = torch.relu(x)
    x = self.conv2(x)
    x = torch.relu(x)
    x = self.conv3(x)
    x = torch.relu(x)
    x = torch.flatten(x)
    x = self.fc(x)
    return x

class AI:
  def __init__(self):
    if os.path.isfile(MODEL):
      self.qnet = torch.load(MODEL)
    else:
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
    game = 1
    if os.path.isfile(LOSSES_CSV):
      with open(LOSSES_CSV, 'r') as losses_csv:
        for row in losses_csv:
          game += 1

    optimizer = optim.Adam(self.qnet.parameters(), LEARNING_RATE)

    with open(LOSSES_CSV, 'a') as losses_csv:
      loss_writer = csv.writer(losses_csv)

      while True:
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
            reward = LIVING_REWARD + GAMMA * qvals[action]
            losses.append(self.update(state_pp, action_pp, reward, optimizer))

          if np.random.rand() < EPSILON:
            action = self.random_move(board)

          board.place(action)

          state_pp = state_p
          action_pp = action_p
          state_p = state
          action_p = action

        if board.winner == 3:
          losses.append(self.update(state_p, action_p, 0.0, optimizer))
          losses.append(self.update(state_pp, action_pp, 0.0, optimizer))
        else:
          losses.append(self.update(state_p, action_p, WIN_REWARD, optimizer))
          losses.append(self.update(state_pp, action_pp, -WIN_REWARD, optimizer))

        loss = np.mean(losses)
        loss_writer.writerow([loss])
        torch.save(self.qnet, MODEL)
        print(f'game: {game}, winner: {int(board.winner)}, loss: {loss}')
        game += 1

  def plot(self):
    losses = []
    if os.path.isfile(LOSSES_CSV):
      with open(LOSSES_CSV, 'r') as losses_csv:
        for row in losses_csv:
          loss = eval(row)
          losses.append(loss)
      plt.figure()
      plt.title('Loss vs. Game')
      plt.plot(range(len(losses)), losses)
      plt.xlabel('Game')
      plt.ylabel('Loss')
      plt.savefig(LOSSES_PLOT)
    