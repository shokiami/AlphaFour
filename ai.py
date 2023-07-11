from game import init_state, get_valid_actions, get_next_state, is_terminal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import matplotlib.pyplot as plt

MODEL_PATH = 'model.pt'
LOSSES_CSV = 'losses.csv'
LOSSES_PLOT = 'losses.png'

NUM_BLOCKS = 4
NUM_CHANNELS = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MCTS_ITRS = 100
UCB_C = 2.0

GAMES_PER_ITR = 100

EPOCHS_PER_ITR = 10
BATCH_SIZE = 64

NUM_ITRS = 1

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
  
# class ConvNet(nn.Module):
#   def __init__(self):
#     super(ConvNet, self).__init__()
#     self.conv1 = nn.Conv2d(3, 32, kernel_size=4, padding=1)
#     self.bn1 = nn.BatchNorm2d(32)
#     self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=1)
#     self.bn2 = nn.BatchNorm2d(64)
#     self.conv3 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
#     self.bn3 = nn.BatchNorm2d(128)
#     self.fc = nn.Linear(1536, 1536)

#     self.policy_head = nn.Linear(1536, 7)
#     self.value_head = nn.Linear(1536, 1)

#     # self.policy_head = nn.Sequential(
#     #   nn.Linear(32 * 42, 7)
#     # )
#     # self.value_head = nn.Sequential(
#     #   nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
#     #   nn.BatchNorm2d(3),
#     #   nn.ReLU(),
#     #   nn.Flatten(),
#     #   nn.Linear(3 * 42, 1)
#     # )

#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = F.relu(x)
#     x = self.conv2(x)
#     x = self.bn2(x)
#     x = F.relu(x)
#     x = self.conv3(x)
#     x = self.bn3(x)
#     x = F.relu(x)
#     x = torch.flatten(x, 1)
#     x = self.fc(x)
#     x = F.relu(x)
#     policy = self.policy_head(x)
#     value = self.value_head(x)
#     return policy, value

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
    return q_val + UCB_C * self.prior * np.sqrt(self.parent.visit_count / (self.visit_count + 1.0))

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
    states = np.stack(states)
    encoded_states = np.stack((states == 1, states == 0, states == -1)).swapaxes(0, 1)
    return torch.tensor(encoded_states, dtype=torch.float32)

  def predict(self, states):
    with torch.no_grad():
      policies, values = self.model(self.to_tensor(states))
    policies = torch.softmax(policies, 1).numpy()
    values = torch.tanh(values).numpy()
    for game in range(len(states)):
      policies[game][np.invert(get_valid_actions(states[game]))] = 0.0
      policies[game] /= np.sum(policies[game])
    return policies, values

  def mcts_search(self, states):
    roots = [MCTSNode(state.copy()) for state in states]
    for i in range(MCTS_ITRS):
      leafs = []
      for game in range(len(states)):
        node = roots[game]
        while len(node.children) > 0:
          node = node.select()
        leafs.append(node)
      policies, values = self.predict([leaf.state for leaf in leafs])
      for game in range(len(states)):
        terminal, win = is_terminal(leafs[game].state, leafs[game].prev_action)
        if terminal:
         leafs[game].backpropagate(-1.0) if win else leafs[game].backpropagate(0.0)
        else:
          leafs[game].expand(policies[game])
          leafs[game].backpropagate(values[game])
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
      input_states = [player * state.copy() for state in states]
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
      print(f'move: {move + 1}, remaining: {len(states)}')
      player = -player
      move += 1
    return examples

  def train(self, examples):
    losses = []
    if os.path.isfile(LOSSES_CSV):
      with open(LOSSES_CSV, 'r') as losses_csv:
        for row in losses_csv:
          losses.append(eval(row))
    with open(LOSSES_CSV, 'a') as losses_csv:
      loss_writer = csv.writer(losses_csv)
      self.model.train()
      for epoch in range(EPOCHS_PER_ITR):
        np.random.shuffle(examples)
        losses = []
        for i in range(0, len(examples), BATCH_SIZE):
          states, policies, values = zip(*examples[i:min(i + BATCH_SIZE, len(examples) - 1)])
          policies = torch.tensor(np.array(policies), dtype=torch.float32)
          values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
          pred_policies, pred_values = self.model(self.to_tensor(states))
          loss = F.cross_entropy(pred_policies, policies) + F.mse_loss(pred_values, values)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          losses.append(loss.item())
        loss_writer.writerow([np.mean(losses)])
        print(f'epoch: {epoch + 1}/{EPOCHS_PER_ITR}')
      self.model.eval()
    self.plot()

  def learn(self):
    for i in range(NUM_ITRS):
      examples = self.self_play()
      self.train(examples)
      torch.save(self.model.state_dict(), MODEL_PATH)
      print(f'iteration: {i + 1}/{NUM_ITRS}')

  def plot(self):
    losses = []
    if not os.path.isfile(LOSSES_CSV):
      return
    with open(LOSSES_CSV, 'r') as losses_csv:
      for row in losses_csv:
        loss = eval(row)
        losses.append(loss)
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
  state = np.array([
    [ 0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0],
    [ 0,  1,  1,  0, -1,  0,  0],
    [ 1, -1,  1,  1,  1,  0, -1],
    [ 1,  1, -1, -1, -1,  0,  1],
  ])
  print(ai.compute(state))

if __name__ == '__main__':
  main()
