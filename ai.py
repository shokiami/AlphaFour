from game import init_state, get_valid_actions, get_next_state, is_terminal, INPUT_SPACE, ACTION_SPACE
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

UCB_C = 1.41421356237  # sqrt(2)
MCTS_ROLLOUTS = 1000

class MCTSNode:
  def __init__(self, state, parent=None, prev_action=None):
    self.state = state
    self.parent = parent
    self.prev_action = prev_action
    self.children = []
    self.expandable_actions = get_valid_actions(state)
    self.visit_count = 0
    self.value_sum = 0

  def is_fully_expanded(self):
    return not np.any(self.expandable_actions) and len(self.children) > 0

  def select(self):
    return max(self.children, key=lambda c: c.ucb())

  def ucb(self):
    q_val = 0.0 if self.visit_count == 0 else 1.0 - (self.value_sum / self.visit_count + 1.0) / 2.0
    return q_val + UCB_C * np.sqrt(np.log(self.parent.visit_count) / self.visit_count)

  def expand(self):
    action = np.random.choice(np.where(self.expandable_actions)[0])
    self.expandable_actions[action] = 0
    next_state = -get_next_state(self.state, 1, action)
    child = MCTSNode(next_state, self, action)
    self.children.append(child)
    return child

  def simulate(self):
    player = 1
    rollout_state = self.state
    while True:
      action = np.random.choice(np.where(get_valid_actions(rollout_state))[0])
      rollout_state = get_next_state(rollout_state, player, action)
      terminal, win = is_terminal(rollout_state, action)
      if terminal:
        return player if win else 0.0
      player = -player

  def backprop(self, value):
    self.value_sum += value
    self.visit_count += 1
    if self.parent is not None:
      self.parent.backprop(-value)

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    r = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x += r
    x = F.relu(x)
    return x

class QNet(nn.Module):
  def __init__(self, num_blocks, num_channels):
    super(QNet, self).__init__()
    self.start_block = nn.Sequential(
      nn.Conv2d(2, num_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_channels),
      nn.ReLU()
    )
    self.res_blocks = [ResBlock(num_channels, num_channels) for i in range(num_blocks)]
    self.policy_head = nn.Sequential(
      nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(32 * INPUT_SPACE, ACTION_SPACE)
    )
    self.value_head = nn.Sequential(
      nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
      nn.BatchNorm2d(3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(3 * INPUT_SPACE, 1),
      nn.Tanh()
    )

  def forward(self, x):
    x = self.start_block(x)
    for res_block in self.res_blocks:
      x = res_block(x)
    policy = self.policy_head(x)
    value = self.value_head(x)
    return policy, value

class AI:
  def __init__(self):
    if os.path.isfile(MODEL):
      self.qnet = torch.load(MODEL)
    else:
      self.qnet = QNet(9, 128)

  def mcts_search(self, state):
    root = MCTSNode(state)
    for i in range(MCTS_ROLLOUTS):
      node = root
      while node.is_fully_expanded():
        node = node.select()
      terminal, win = is_terminal(node.state, node.prev_action)
      if terminal:
        value = -1.0 if win else 0.0
      else:
        node = node.expand()
        value = node.simulate()
      node.backprop(value)
    action_probs = np.zeros(ACTION_SPACE)
    for child in root.children:
      action_probs[child.prev_action] = child.visit_count
    action_probs /= np.sum(action_probs)
    return action_probs

  def board_to_state(self, board):
    mat = board.mat.copy()
    mat[mat == 2] = -1.0
    if board.player == 2:
      mat *= -1.0
    return torch.FloatTensor(mat).unsqueeze(0)

  def compute_move(self, board):
    return np.random.randint(7), 0
    # state = self.board_to_state(board)
    # with torch.no_grad():
    #   qvals = self.qnet(state).numpy()
    # action = self.safe_argmax(qvals, board)
    # qval = qvals[action]
    # return action, qval

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
    