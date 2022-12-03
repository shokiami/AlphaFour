from board import Board
import numpy as np
import torch
from torch import nn
from torch import optim

NUM_GAMES = 100000
LEARNING_RATE = 0.001
EPSILON = 0.01
GAMMA = 0.99
REWARD = 1000

class QNet(nn.Module):
  def __init__(self):
    super(QNet, self).__init__()
    self.conv = nn.Conv2d(1, 32, 4, 1, 1)
    self.fc = nn.Linear(960, 7)

  def forward(self, x):
    x = self.conv(x)
    x = torch.relu(x)
    x = torch.flatten(x)
    x = self.fc(x)
    return x

class AI:
  def __init__(self):
    self.q_nets = [QNet(), QNet()]
    self.optimizers = [optim.Adam(self.q_nets[0].parameters(), LEARNING_RATE), optim.Adam(self.q_nets[1].parameters(), LEARNING_RATE)]
    self.loss_func = nn.MSELoss()

  def compute(self, board):
    with torch.no_grad():
      q_net = self.q_nets[1]
      q_vals = q_net(self.board_to_tensor(board)).numpy()
      x = np.argmax(q_vals)
      q_val = q_vals[x]
      return x, q_val

  def board_to_tensor(self, board):
    mat = board.mat.copy()
    if board.player == 1:
      mat[mat == 2] = -1
    else:
      mat[mat == 1] = -1
      mat[mat == 2] = 1
    tensor = torch.tensor(mat, dtype=torch.float).unsqueeze(0)
    return tensor

  def train(self):
    for i in range(NUM_GAMES):
      board = Board()
      winner = 0
      losses = []
      while winner == 0:
        q_net1 = self.q_nets[board.player - 1]
        q_net2 = self.q_nets[2 - board.player]
        optimizer = self.optimizers[board.player - 1]
        # get Q-values
        q_vals_tensor = q_net1(self.board_to_tensor(board))
        q_vals = q_vals_tensor.detach().numpy()
        # get action according to epsilon-greedy
        if np.random.rand() < EPSILON:
          x = np.random.randint(7)
        else:
          x = np.argmax(q_vals)
        # get reward
        reward = 0
        if not board.placeable(x):
          reward = -10 * REWARD
        else:
          player = board.player
          winner = board.place(x)
          if winner == player:
            reward = REWARD
        # get new Q-value
        if winner == 0:
          with torch.no_grad():
            next_val = -np.max(q_net2(self.board_to_tensor(board)).numpy())
          new_q_val = reward + GAMMA * next_val
        else:
          new_q_val = reward
        # get new Q-values for state
        new_q_vals = q_vals.copy()
        new_q_vals[x] = new_q_val
        new_q_vals_tensor = torch.tensor(new_q_vals)
        # update QNet
        optimizer.zero_grad()
        loss = self.loss_func(q_vals_tensor, new_q_vals_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
      print(f"game: {i + 1}/{NUM_GAMES}, loss: {np.mean(losses)}")
    print("done training!")

  def test(self):
    TEST_GAMES = 100
    for i in range(TEST_GAMES):
      board = Board()
      winner = 0
      wins = 0
      while winner == 0:
        x = self.compute(board)[0]
        if not board.placeable(x):
          break
        player = board.player
        winner = board.place(x)
        if winner == player:
          wins += 1
          break
        # opponent moves randomly
        x = np.random.randint(7)
        while not board.placeable(x):
          x = np.random.randint(7)
        player = board.player
        winner = board.place(x)
    print(f"win rate: {wins}/{TEST_GAMES}")
