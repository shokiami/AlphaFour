from board import Board
import numpy as np
import torch
from torch import nn
from torch import optim

NUM_GAMES = 10000
LEARNING_RATE = 0.01
EPSILON = 0.1
GAMMA = 0.99
REWARD = 1000

class QNet(nn.Module):
  def __init__(self):
    super(QNet, self).__init__()
    self.conv = nn.Conv2d(1, 128, 4, 1, 1)
    self.fc = nn.Linear(3840, 7)

  def forward(self, x):
    x = self.conv(x)
    x = torch.relu(x)
    x = torch.flatten(x)
    x = self.fc(x)
    return x

class AI:
  def __init__(self):
    self.q_net = QNet()

  def compute(self, board):
    with torch.no_grad():
      q_vals = self.q_net(self.board_to_tensor(board)).numpy()
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
    optimizer = optim.Adam(self.q_net.parameters(), LEARNING_RATE)
    loss_func = nn.MSELoss()
    for i in range(NUM_GAMES):
      board = Board()
      winner = 0
      losses = []
      while winner == 0:
        # get action according to epsilon-greedy
        if np.random.rand() < EPSILON:
          x = np.random.randint(7)
          while not board.placeable(x):
            x = np.random.randint(7)
        else:
          x = self.compute(board)[0]
        # get Q-values
        q_vals_tensor = self.q_net(self.board_to_tensor(board))
        q_vals = q_vals_tensor.detach().numpy()
        # get reward
        new_q_val = 0
        if not board.placeable(x):
          new_q_val = -REWARD
        else:
          winner = board.place(x)
          if winner == board.player:
            new_q_val = REWARD
          elif winner == 3 - board.player:
            new_q_val = -REWARD
          else:
            # get value of next state
            with torch.no_grad():
              next_val = -np.max(self.q_net(self.board_to_tensor(board)).numpy())
            # get new Q-value for action
            new_q_val = GAMMA * next_val
        # get new Q-values for state
        new_q_vals = q_vals.copy()
        new_q_vals[x] = new_q_val
        new_q_vals_tensor = torch.tensor(new_q_vals)
        # perform gradient descent
        optimizer.zero_grad()
        loss = loss_func(q_vals_tensor, new_q_vals_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
      print(f"game: {i + 1}/{NUM_GAMES}, loss: {np.mean(losses)}")
    print("done training!")
