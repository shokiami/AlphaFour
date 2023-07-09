from game import init_state, get_valid_actions, get_next_state, is_terminal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import matplotlib.pyplot as plt

NUM_BLOCKS = 4
NUM_CHANNELS = 64

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MODEL_PATH = 'model.pt'
# LOSSES_CSV = 'losses.csv'
# LOSSES_PLOT = 'losses.png'

TRAIN_MCTS_ITRS = 200
TEST_MCTS_ITRS = 1000
NUM_ITRS = 1
GAMES_PER_ITR = 200
EPOCHS_PER_ITR = 16
BATCH_SIZE = 32

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
      nn.Linear(32 * 42, 7)
    )
    self.value_head = nn.Sequential(
      nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
      nn.BatchNorm2d(3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(3 * 42, 1),
      nn.Tanh()
    )

  def forward(self, x):
    x = self.start_block(x)
    for res_block in self.res_blocks:
      x = res_block(x)
    policy = self.policy_head(x)
    value = self.value_head(x)
    return policy, value

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
    q_val = 0.0 if self.visit_count == 0 else 0.5 - 0.5 * self.value_sum / self.visit_count
    return q_val + 2.0 * self.prior * np.sqrt(self.parent.visit_count / (self.visit_count + 1.0))

  def expand(self, policy):
    for action, prob in enumerate(policy):
      if prob > 0.0:
        next_state = -get_next_state(self.state, 1, action)
        child = MCTSNode(next_state, self, action, prob)
        self.children.append(child)

  def backpropagate(self, value):
    self.value_sum += value
    self.visit_count += 1
    if self.parent is not None:
      self.parent.backpropagate(-value)

class AI:
  def __init__(self):
    self.model = ResNet(NUM_BLOCKS, NUM_CHANNELS)
    if os.path.isfile(MODEL_PATH):
      self.model.load_state_dict(torch.load(MODEL_PATH))
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    self.model.eval()

  def to_tensor(self, states):
    return torch.tensor(np.stack((states == 1, states == 0, states == -1)).swapaxes(0, 1), dtype=torch.float32)

  def parallel_predict(self, states):
    with torch.no_grad():
      policies, values = self.model(self.to_tensor(states))
    policies = torch.softmax(policies, axis=1).numpy()
    values = values.numpy()
    for game in range(len(states)):
      policies[game][np.invert(get_valid_actions(states[game]))] = 0.0
      policies[game] /= np.sum(policies[game])
    return policies, values

  def predict(self, state):
    with torch.no_grad():
      policy, value = self.model(self.to_tensor(np.expand_dims(state, 0)))
    policy = torch.softmax(policy, axis=1).squeeze(0).numpy()
    policy[np.invert(get_valid_actions(state))] = 0.0
    policy /= np.sum(policy)
    value = value.item()
    return policy, value

  def parallel_mcts_search(self, states):
    roots = [MCTSNode(state.copy()) for state in states]
    for i in range(TRAIN_MCTS_ITRS):
      leafs = []
      leaf_states = []
      for game in range(len(states)):
        node = roots[game]
        while len(node.children) > 0:
          node = node.select()
        leafs.append(node)
        leaf_states.append(node.state)
      policies, values = self.parallel_predict(np.array(leaf_states))
      for game in range(len(states)):
        leaf = leafs[game]
        terminal, win = is_terminal(leaf.state, leaf.prev_action)
        if terminal:
          value = -1.0 if win else 0.0
        else:
          policy = policies[game]
          value = values[game]
          leaf.expand(policy)
        leaf.backpropagate(value)
      print(i)
    policies = []
    for game in range(len(states)):
      policy = np.zeros(7)
      for child in roots[game].children:
        policy[child.prev_action] = child.visit_count
      policies.append(policy / np.sum(policy))
    return policies

  def mcts_search(self, state):
    root = MCTSNode(state.copy())
    for i in range(TEST_MCTS_ITRS):
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
    policy = np.zeros(7)
    for child in root.children:
      policy[child.prev_action] = child.visit_count
    policy /= np.sum(policy)
    return policy

  def self_play(self):
    examples = []
    player = 1
    states = GAMES_PER_ITR * [init_state()]
    while len(states) > 0:
      policies = self.parallel_mcts_search([player * state for state in states])
      curr_examples = []
      for game in reversed(range(len(states))):
        action = np.random.choice(7, p=policies[game])
        states[game] = get_next_state(states[game], player, action)
        curr_examples.append([player * states[game], policies[game], 0.0])
        terminal, win = is_terminal(states[game], action)
        if terminal:
          print(states[game])
          if win:
            for i in range(len(curr_examples)):
              curr_examples[i][2] = 1.0 if (len(curr_examples) - i) % 2 == 1 else -1.0
          examples += curr_examples
          states.pop(game)
      player = -player
      print(f'games remaining: {len(states)}')
    return examples

  def train(self, examples):
    self.model.train()
    for epoch in range(EPOCHS_PER_ITR):
      np.random.shuffle(examples)
      for i in range(0, len(examples), BATCH_SIZE):
        states, policies, values = zip(*examples[i:min(i + BATCH_SIZE, len(examples))])
        states = self.to_tensor(np.array(states))
        policies = torch.tensor(np.array(policies), dtype=torch.float32)
        values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
        pred_policies, pred_values = self.model(states)
        loss = F.cross_entropy(pred_policies, policies) + F.mse_loss(pred_values, values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      print(f'epoch: {epoch + 1}/{EPOCHS_PER_ITR}')
    self.model.eval()

  def learn(self):
    for i in range(NUM_ITRS):
      examples = self.self_play()
      self.train(examples)
      # torch.save(self.model.state_dict(), MODEL_PATH)
      print(f'iteration: {i + 1}/{NUM_ITRS}')

  def compute(self, state):
    policy = self.mcts_search(-state)
    print(policy)
    return np.argmax(policy)
