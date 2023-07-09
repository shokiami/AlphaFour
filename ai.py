from game import init_state, get_valid_actions, get_next_state, is_terminal, INPUT_SPACE, ACTION_SPACE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import csv
import matplotlib.pyplot as plt

NUM_HIDDEN = 4
NUM_CHANNELS = 64


LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MODEL_PATH = 'model.pt'
# LOSSES_CSV = 'losses.csv'
# LOSSES_PLOT = 'losses.png'

UCB_C = 2.0
MCTS_ROLLOUTS = 10

NUM_ITRS = 2
GAMES_PER_ITR = 10
NUM_EPOCHS = 8
BATCH_SIZE = 4

NUM_PARALLEL_GAMES = 100

class MCTSNode:
  def __init__(self, state, parent=None, prev_action=None, prior=0):
    self.state = state
    self.parent = parent
    self.prev_action = prev_action
    self.prior = prior
    self.children = []
    self.visit_count = 0
    self.value_sum = 0

  def select(self):
    return max(self.children, key=lambda c: c.ucb())

  def ucb(self):
    q_val = 0.0 if self.visit_count == 0 else 1.0 - (self.value_sum / self.visit_count + 1.0) / 2.0
    return q_val + UCB_C * self.prior * np.sqrt(self.parent.visit_count / (self.visit_count + 1.0))

  def expand(self, policy):
    for action, prob in enumerate(policy):
      if prob > 0:
        next_state = -get_next_state(self.state, 1, action)
        child = MCTSNode(next_state, self, action, prob)
        self.children.append(child)

  def backpropagate(self, value):
    self.value_sum += value
    self.visit_count += 1
    if self.parent is not None:
      self.parent.backpropagate(-value)

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

class ResNet(nn.Module):
  def __init__(self, num_blocks, num_channels):
    super(ResNet, self).__init__()
    self.start_block = nn.Sequential(
      nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
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
    self.model = ResNet(NUM_HIDDEN, NUM_CHANNELS)
    if os.path.isfile(MODEL_PATH):
      self.model.load_state_dict(torch.load(MODEL_PATH))
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

  def to_tensor(self, states):
    return torch.tensor(np.stack((states == 1, states == 0, states == -1)).swapaxes(0, 1), dtype=torch.float32)

  def predict(self, state):
    with torch.no_grad():
      policy, value = self.model(self.to_tensor(np.expand_dims(state, 0)))
    policy = torch.softmax(policy, axis=1).squeeze(0).numpy()
    policy[np.invert(get_valid_actions(state))] = 0.0
    policy /= np.sum(policy)
    value = value.item()
    return policy, value

  def mcts_search(self, state):
    self.model.eval()
    root = MCTSNode(state.copy())
    for i in range(MCTS_ROLLOUTS):
      node = root
      while len(node.children) > 0:
        node = node.select()
      terminal, win = is_terminal(node.state, node.prev_action)
      if terminal:
        value = -1.0 if win else 0.0
      else:
        policy, value = self.predict(node.state)
        node.expand(policy)
      node.backpropagate(value)
    action_probs = np.zeros(ACTION_SPACE)
    for child in root.children:
      action_probs[child.prev_action] = child.visit_count
    action_probs /= np.sum(action_probs)
    return action_probs

  def self_play(self):
    examples = []
    player = 1
    state = init_state()
    while True:
      action_probs = self.mcts_search(player * state) ** (1 / 1.25)
      action_probs /= action_probs.sum()
      action = np.random.choice(7, p=action_probs)
      state = get_next_state(state, player, action)
      examples.append([player * state, action_probs, 0.0])
      terminal, win = is_terminal(state, action)
      if terminal:
        if win:
          for i in range(len(examples)):
            examples[i][2] = 1.0 if (len(examples) - i) % 2 == 1 else -1.0
        return examples
      player = -player

  def train(self, examples):
    self.model.train()
    np.random.shuffle(examples)
    for i in range(0, len(examples), BATCH_SIZE):
      states, action_probs, values = zip(*examples[i:min(i + BATCH_SIZE, len(examples))])
      states = self.to_tensor(np.array(states))
      action_probs = torch.tensor(np.array(action_probs), dtype=torch.float32)
      values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
      pred_policies, pred_values = self.model(states)
      # print(pred_policies, action_probs)
      # print(pred_values, values)
      loss = F.cross_entropy(pred_policies, action_probs) + F.mse_loss(pred_values, values)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      print(loss.item())

  def learn(self):
    for i in range(NUM_ITRS):
      examples = []
      for j in range(GAMES_PER_ITR):
        examples += self.self_play()
        print(j)
      for epoch in range(NUM_EPOCHS):
        self.train(examples)
        print(epoch)
      # torch.save(self.model.state_dict(), MODEL_PATH)
      print(i)

  def compute(self, state):
    action_probs = self.mcts_search(-state)
    print(action_probs)
    return np.argmax(action_probs)
