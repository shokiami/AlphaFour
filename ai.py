from game import init_state, get_valid_actions, get_next_state, is_terminal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

MODEL_PATH = 'model.pt'
LOSSES_PLOT = 'losses.png'
NUM_BLOCKS = 4
NUM_CHANNELS = 64
LEARNING_RATE = 0.001
MCTS_ITRS = 100
GAMES_PER_ITR = 100
EPOCHS_PER_ITR = 10
BATCH_SIZE = 64
NUM_ITRS = 1

class ResBlock(nn.Module):
  def __init__(self, num_channels):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(num_channels)

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
    self.res_blocks = nn.ModuleList([ResBlock(num_channels) for i in range(num_blocks)])
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
    return max(self.children, key=lambda child: child.ucb())

  def ucb(self):
    q_val = 0.0 if self.visit_count == 0 else 0.5 - 0.5 * self.value_sum / self.visit_count
    return q_val + 2.0 * self.prior * np.sqrt(self.parent.visit_count) / (self.visit_count + 1.0)

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
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

  def to_tensor(self, states):
    states = np.stack(states)
    encoded_states = np.stack((states == 1, states == 0, states == -1)).swapaxes(0, 1)
    return torch.tensor(encoded_states, dtype=torch.float32)

  def predict(self, states):
    with torch.no_grad():
      policies, values = self.model(self.to_tensor(states))
    policies = torch.softmax(policies, 1).numpy()
    values = values.numpy()
    for game in range(len(states)):
      policies[game][~get_valid_actions(states[game])] = 0.0
      policies[game] /= np.sum(policies[game])
    return policies, values

  def mcts_search(self, states):
    self.model.eval()
    roots = [MCTSNode(state) for state in states]
    for i in range(MCTS_ITRS):
      leafs = []
      for game in range(len(states)):
        node = roots[game]
        while len(node.children) > 0:
          node = node.select()
        terminal, win = is_terminal(node.state, node.prev_action)
        if terminal:
          node.backpropagate(-1.0) if win else node.backpropagate(0.0)
        else:
          leafs.append(node)
      if len(leafs) > 0:
        policies, values = self.predict([leaf.state for leaf in leafs])
        for j in range(len(leafs)):
          leafs[j].expand(policies[j])
          leafs[j].backpropagate(values[j])
    policies = []
    for game in range(len(states)):
      policy = np.zeros(7)
      for child in roots[game].children:
        policy[child.prev_action] = child.visit_count
      policy /= np.sum(policy)
      policies.append(policy)
    return policies

  def self_play(self):
    examples = []
    player = 1
    states = [init_state() for game in range(GAMES_PER_ITR)]
    curr_examples = [[] for game in range(GAMES_PER_ITR)]
    move = 0
    while len(states) > 0:
      input_states = [player * state for state in states]
      policies = self.mcts_search(input_states)
      for game in reversed(range(len(states))):
        curr_examples[game].append([input_states[game], policies[game], 0.0])
        action = np.random.choice(7, p=policies[game])
        states[game] = get_next_state(states[game], player, action)
        terminal, win = is_terminal(states[game], action)
        if terminal:
          print(states[game])
          if win:
            for i in range(len(curr_examples[game])):
              curr_examples[game][i][2] = 1.0 if (len(curr_examples[game]) - i) % 2 == 1 else -1.0
          examples += curr_examples[game]
          states.pop(game)
          curr_examples.pop(game)
      print(f'move: {move + 1}, remaining: {len(states)}')
      player = -player
      move += 1
    return examples

  def train(self, examples):
    self.model.train()
    np.random.shuffle(examples)
    losses = []
    for i in range(0, len(examples), BATCH_SIZE):
      states, policies, values = zip(*examples[i: i + BATCH_SIZE])
      policies = torch.tensor(np.array(policies), dtype=torch.float32)
      values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
      pred_policies, pred_values = self.model(self.to_tensor(states))
      loss = F.cross_entropy(pred_policies, policies) + F.mse_loss(pred_values, values)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      losses.append(loss.item())
    return np.mean(losses)

  def learn(self):
    losses = []
    for i in range(NUM_ITRS):
      examples = self.self_play()
      for epoch in range(EPOCHS_PER_ITR):
        loss = self.train(examples)
        losses.append(loss)
        print(f'epoch: {epoch + 1}/{EPOCHS_PER_ITR}')
      torch.save(self.model.state_dict(), MODEL_PATH)
      self.plot(losses)
      print(f'iteration: {i + 1}/{NUM_ITRS}')

  def plot(self, losses):
    plt.figure()
    plt.title('Loss vs. Epoch')
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(LOSSES_PLOT)

  def compute(self, state):
    policy = self.mcts_search([-state])[0]
    print(policy.round(4))
    return np.argmax(policy)

def main():
  ai = AI()
  ai.learn()

if __name__ == '__main__':
  main()
