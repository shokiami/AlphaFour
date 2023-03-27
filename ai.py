from board import Board
import numpy as np
import torch
from torch import nn
from torch import optim

NUM_GAMES = 100
LEARNING_RATE = 0.0001
EPSILON = 0.1
GAMMA = 0.9

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
    self.qnet = QNet()

  def board_to_tensor(self, board):
    return torch.FloatTensor(board.mat.copy()).unsqueeze(0)

  def random_move(self, board):
    x = np.random.randint(7)
    while not board.placeable(x):
      x = np.random.randint(7)
    return x

  def compute_move(self, board):
    with torch.no_grad():
      qvals = self.qnet(self.board_to_tensor(board)).numpy()
    x = np.argmax(qvals)
    qval = qvals[x]
    return x, qval

    # return self.minimax(board, board.player, depth, -np.inf, np.inf)

  # def minimax(self, board, player, depth, alpha, beta):
  #   best_x = None
  #   if depth == 0 or board.winner != 0:
  #     val_net = self.val_nets[player]
  #     with torch.no_grad():
  #       val = val_net(self.board_to_tensor(board)).item()
  #     return best_x, val
  #   my_turn = (board.player == player)
  #   val = -np.inf if my_turn else np.inf
  #   for x in range(7):
  #     if board.placeable(x):
  #       board.place(x)
  #       next_val = self.minimax(board, player, depth - 1, alpha, beta)[1]
  #       board.undo()
  #       if my_turn:
  #         if next_val > val:
  #           val = next_val
  #           best_x = x
  #         if val > beta:
  #           return best_x, val
  #         alpha = max(alpha, val)
  #       else:
  #         if next_val < val:
  #           val = next_val
  #           best_x = x
  #         if val < alpha:
  #           return best_x, val
  #         beta = min(beta, val)
  #   return best_x, val

  def train(self):
    loss_func = nn.MSELoss()
    adam = optim.Adam(self.qnet.parameters(), LEARNING_RATE)
    for i in range(NUM_GAMES):
      board = Board()
      losses = []

      board.place(self.random_move(board))

      while board.winner == 0:
        qvals_tensor = self.qnet(self.board_to_tensor(board))
        qvals = qvals_tensor.detach().numpy()
        new_qvals = qvals.copy()

        if np.random.rand() < EPSILON:
          x = np.random.randint(7)
        else:
          x = np.argmax(qvals)

        if not board.placeable(x):
          new_qvals[x] = -1000.0
        else:
          board.place(x)
          if board.winner == 2:
            reward = 100.0
          elif board.winner ==3:
            reward = 0.0
          else:
            board.place(self.random_move(board))
            if board.winner == 1:
              reward = -100.0
            else:
              reward = 1.0
            with torch.no_grad():
              next_qvals = self.qnet(self.board_to_tensor(board)).numpy()
          new_qvals[x] = reward + GAMMA * np.max(next_qvals)

        new_qvals_tensor = torch.tensor(new_qvals)

        adam.zero_grad()
        loss = loss_func(qvals_tensor, new_qvals_tensor)
        loss.backward()
        adam.step()
        losses.append(loss.item())

      print(f'game: {i + 1}/{NUM_GAMES}, winner: {int(board.winner)}, losses: {np.mean(losses)}')
    print('done training!')

  # def test(self):
  #   TEST_GAMES = 100
  #   wins = 0
  #   for i in range(TEST_GAMES):
  #     board = Board()
  #     while board.winner == 0:
  #       if board.player == 1:
  #         x = self.random_move(board)
  #       else:
  #         x = self.compute_move(board)[0]
  #       board.place(x)
  #     if board.winner == 2:
  #       wins += 1
  #   print(f'win rate: {wins}/{TEST_GAMES}')
