from re import X
from board import Board
import torch
from torch import nn
from torch import optim

class ConvBlock(nn.Module):
  def __init__(self):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv2d(1, 128, 3, padding=1)
    self.bn = nn.BatchNorm2d(128)

  def forward(self, x):
    x = self.conv(x)
    x = torch.relu(self.bn(x))
    return x

class ResBlock(nn.Module):
  def __init__(self):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(128)
    self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(128)

  def forward(self, x):
    res = x
    x = self.conv1(x)
    x = torch.relu(self.bn1(x))
    x = self.conv2(x)
    x = self.bn2(x)
    x += res
    x = torch.relu(x)
    return x

class OutBlock(nn.Module):
  def __init__(self):
    super(OutBlock, self).__init__()
    self.conv = nn.Conv2d(128, 1, 1)
    self.fc1 = nn.Linear(6 * 7, 32)
    self.fc2 = nn.Linear(32, 1)

  def forward(self, x):
    x = self.conv(x)
    x = torch.relu(self.fc1(x.flatten()))
    x = torch.sigmoid(self.fc2(x))
    return x

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.conv = ConvBlock()
    self.res1 = ResBlock()
    self.res2 = ResBlock()
    self.res3 = ResBlock()
    self.out = OutBlock()

  def forward(self, mat):
    x = torch.from_numpy(mat)
    x = x.unsqueeze(dim=0).unsqueeze(dim=0).float()
    x = self.conv(x)
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.out(x)
    return x

class AI:
  def __init__(self):
    self.model = Model()

  def predict(self, board, depth):
    if depth == 0:
      print("ERROR: Depth must be >= 1.")
      return
    best_x = 0
    best_val = 0
    for x in range(7):
      if board.placeable(x):
        board.place(x)
        if depth == 1:
          with torch.no_grad():
            val = self.model(board.mat).item()
        else:
          val = 1 - self.predict(board, depth - 1)[1]
        if val > best_val:
          best_x = x
          best_val = val
        board.undo()
    return best_x, best_val

  def compute(self, board):
    depth = 3
    best_x, best_val = self.predict(board, depth)
    print(best_val)
    return best_x
  
  def train(self):
    optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for i in range(100):
      board = Board()
      winner = 0
      history = {"player 1": [], "player 2": []}
      while winner == 0:
        player = board.player
        x = self.predict(board, 1)[0]
        winner = board.place(x)
        history[f"player {player}"].append(board.mat.copy())
      for player in [1, 2]:
        if winner == player:
          val = 1
        elif winner == 3:
          val = 0.5
        else:
          val = 0
        for mat in history[f"player {player}"]:
          optimizer.zero_grad()
          val_hat = self.model(mat)
          loss = criterion(val_hat, torch.tensor([val]).float())
          loss.backward()
          optimizer.step()
      print(f"game: {i + 1}")
    print("done training!")
