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
    mat = board.mat.copy()
    mat[mat == 2] = -1.0
    if board.player == 2:
      mat *= -1.0
    return torch.FloatTensor(mat).unsqueeze(0)

  def compute_move(self, board):
    with torch.no_grad():
      qvals = self.qnet(self.board_to_tensor(board)).numpy()
    x = self.safe_argmax(qvals, board)
    qval = qvals[x]
    return x, qval

  def safe_argmax(self, qvals, board):
    for x in np.argsort(qvals):
      if board.placeable(x):
        return x

  def random_move(self, board):
    x = np.random.randint(7)
    while not board.placeable(x):
      x = np.random.randint(7)
    return x

  def update(self, board_tensor, x, reward, optimizer):
    qvals_tensor = self.qnet(board_tensor)
    new_qvals_tensor = qvals_tensor.detach().clone()
    new_qvals_tensor[x] = reward
    optimizer.zero_grad()
    loss = F.mse_loss(qvals_tensor, new_qvals_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

  def train(self):
    self.qnet.train()
    optimizer = optim.Adam(self.qnet.parameters(), LEARNING_RATE)
    for i in range(NUM_GAMES):
      board = Board()
      losses = []

      board_tensor_p = None
      x_p = None
      board_tensor_pp = None
      x_pp = None

      while board.winner == 0:
        board_tensor = self.board_to_tensor(board)
        with torch.no_grad():
          qvals_tensor = self.qnet(board_tensor)
        qvals = qvals_tensor.numpy()

        if x_pp:
          reward = 1.0 + GAMMA * qvals[self.safe_argmax(qvals, board)]
          losses.append(self.update(board_tensor_pp, x_pp, reward, optimizer))

        if np.random.rand() < EPSILON:
          x = self.random_move(board)
        else:
          x = self.safe_argmax(qvals, board)

        board.place(x)

        board_tensor_pp = board_tensor_p
        x_pp = x_p
        board_tensor_p = board_tensor
        x_p = x

      if board.winner == 3:
        losses.append(self.update(board_tensor_pp, x_pp, 0.0, optimizer))
        losses.append(self.update(board_tensor_p, x_p, 0.0, optimizer))
      else:
        losses.append(self.update(board_tensor_pp, x_pp, -100.0, optimizer))
        losses.append(self.update(board_tensor_p, x_p, 100.0, optimizer))

      print(f'game: {i + 1}/{NUM_GAMES}, winner: {int(board.winner)}, loss: {np.mean(losses)}')

    self.qnet.eval()
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
