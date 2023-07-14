import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

MODELS = 'models'
NUM_BLOCKS = 8
NUM_CHANNELS = 128
LEARNING_RATE = 0.001
MCTS_ITRS = 100
C_PUCT = 2.0
DIRICHLET_EPSILON = 0.1
DIRICHLET_ALPHA = 1.4

class ResBlock(nn.Module):
  def __init__(self, num_channels):
    super(ResBlock, self).__init__()
    self.conv_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    self.bn_1 = nn.BatchNorm2d(num_channels)
    self.conv_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    self.bn_2 = nn.BatchNorm2d(num_channels)

  def forward(self, x):
    r = x
    x = self.conv_1(x)
    x = self.bn_1(x)
    x = F.relu(x)
    x = self.conv_2(x)
    x = self.bn_2(x)
    x += r
    x = F.relu(x)
    return x

class ResNet(nn.Module):
  def __init__(self, game, num_blocks, num_channels):
    super(ResNet, self).__init__()
    self.game = game
    self.start_block = nn.Sequential(
      nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_channels),
      nn.ReLU()
    )
    self.res_blocks = nn.ModuleList([ResBlock(num_channels) for i in range(num_blocks)])
    self.policy_head = nn.Sequential(
      nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(32 * self.game.state_size, self.game.action_size)
    )
    self.value_head = nn.Sequential(
      nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
      nn.BatchNorm2d(3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(3 * self.game.state_size, 1),
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
  def __init__(self, game, state, parent=None, prev_action=None, prior=0):
    self.game = game
    self.state = state
    self.parent = parent
    self.prev_action = prev_action
    self.prior = prior
    self.children = []
    self.value_sum = 0.0
    self.visit_count = 0

  def select(self):
    return max(self.children, key=lambda child: child.ucb())

  def ucb(self):
    exploit = 0.0 if self.visit_count == 0 else 0.5 - 0.5 * self.value_sum / self.visit_count
    explore = C_PUCT * self.prior * np.sqrt(self.parent.visit_count) / (self.visit_count + 1)
    return exploit + explore

  def expand(self, policy):
    for i in range(len(policy)):
      if policy[i] > 0.0:
        child_state = -self.game.next_state(self.state, 1, i)
        child = MCTSNode(self.game, child_state, self, i, policy[i])
        self.children.append(child)

  def backprop(self, value):
    self.value_sum += value
    self.visit_count += 1
    if self.parent is not None:
      self.parent.backprop(-value)

class AlphaFour:
  def __init__(self, game, gen):
    self.game = game
    self.model = ResNet(game, NUM_BLOCKS, NUM_CHANNELS)
    model_path = os.path.join(MODELS, f'model_{gen}.pt')
    if os.path.exists(model_path) or gen > 0:
      self.model.load_state_dict(torch.load(model_path))
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

  def to_tensor(self, states):
    states = np.stack(states)
    encoded_states = np.stack((states == 1, states == 0, states == -1)).swapaxes(0, 1)
    return torch.tensor(encoded_states, dtype=torch.float32)

  def mcts(self, state):
    policies, values = self.mcts_parallel([state], False)
    return policies[0], values[0]

  def mcts_parallel(self, states, train=True):
    self.model.eval()
    roots = [MCTSNode(self.game, state) for state in states]
    for i in range(MCTS_ITRS):
      leafs = []
      for root in roots:
        node = root
        while len(node.children) > 0:
          node = node.select()
        terminal, win = self.game.is_terminal(node.state, node.prev_action)
        if terminal:
          node.backprop(-1.0) if win else node.backprop(0.0)
        else:
          leafs.append(node)
      if len(leafs) > 0:
        leaf_states = [leaf.state for leaf in leafs]
        with torch.no_grad():
          policies, values = self.model(self.to_tensor(leaf_states))
        policies = torch.softmax(policies, 1).numpy()
        values = values.squeeze(1).numpy()
        if train and i == 0:
          noise = np.random.dirichlet(self.game.action_size * [DIRICHLET_ALPHA], len(leafs))
          policies = (1.0 - DIRICHLET_EPSILON) * policies + DIRICHLET_EPSILON * noise
        for j in range(len(leafs)):
          policies[j][~self.game.valid_actions(leaf_states[j])] = 0.0
          policies[j] /= np.sum(policies[j])
          leafs[j].expand(policies[j])
          leafs[j].backprop(values[j])
    policies = []
    values = []
    for root in roots:
      policy = np.zeros(self.game.action_size)
      for child in root.children:
        policy[child.prev_action] = child.visit_count
      policy /= np.sum(policy)
      policies.append(policy)
      values.append(root.value_sum / root.visit_count)
    return policies, values
  
  def compute_action(self, state):
    policy, value = self.mcts(-state)
    print(f'policy: {policy.round(2)}')
    print(f'value: {value.round(4)}')
    return np.argmax(policy)
