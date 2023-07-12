from game import ConnectFour
from ai import AlphaFour, MODELS
import numpy as np
import torch
import torch.nn.functional as F
import os
import shutil
import matplotlib.pyplot as plt
import time
from datetime import timedelta

LOSS_PLOT = 'loss.png'
GAMES_PER_GEN = 100
EPOCHS_PER_GEN = 10
BATCH_SIZE = 32
NUM_GENS = 100

def self_play(ai):
  examples = []
  player = 1
  states = [ai.game.init_state() for i in range(GAMES_PER_GEN)]
  curr_examples = [[] for i in range(GAMES_PER_GEN)]
  move = 0
  while len(states) > 0:
    input_states = [player * state for state in states]
    policies = ai.monte_carlo_tree_search(input_states)
    for i in reversed(range(len(states))):
      curr_examples[i].append([input_states[i], policies[i], 0.0])
      action = np.random.choice(ai.game.action_size, p=policies[i])
      states[i] = ai.game.next_state(states[i], player, action)
      terminal, win = ai.game.is_terminal(states[i], action)
      if terminal:
        print(states[i])
        if win:
          for j in range(len(curr_examples[i])):
            curr_examples[i][j][2] = 1.0 if (len(curr_examples[i]) - j) % 2 == 1 else -1.0
        examples += curr_examples[i]
        states.pop(i)
        curr_examples.pop(i)
    player = -player
    print(f'move: {move + 1}, remaining: {len(states)}')
    move += 1
  return examples

def train(ai, examples):
  ai.model.train()
  np.random.shuffle(examples)
  losses = []
  for i in range(0, len(examples), BATCH_SIZE):
    states, policies, values = zip(*examples[i: i + BATCH_SIZE])
    policies = torch.tensor(np.array(policies), dtype=torch.float32)
    values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
    pred_policies, pred_values = ai.model(ai.to_tensor(states))
    loss = F.cross_entropy(pred_policies, policies) + F.mse_loss(pred_values, values)
    ai.optimizer.zero_grad()
    loss.backward()
    ai.optimizer.step()
    losses.append(loss.item())
  return np.mean(losses)

def plot(losses):
  plt.figure()
  plt.title('Loss vs. Epoch')
  plt.plot(range(len(losses)), losses)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig(LOSS_PLOT)

def main():
  start = time.time()
  cf = ConnectFour()
  ai = AlphaFour(cf)
  if os.path.isdir(MODELS):
    shutil.rmtree(MODELS)
  os.makedirs(MODELS)
  torch.save(ai.model.state_dict(), os.path.join(MODELS, 'model_0.pt'))
  losses = []
  for i in range(NUM_GENS):
    examples = self_play(ai)
    for epoch in range(EPOCHS_PER_GEN):
      loss = train(ai, examples)
      losses.append(loss)
      print(f'epoch: {epoch + 1}/{EPOCHS_PER_GEN}')
    plot(losses)
    if (i + 1) % 10 == 0:
      torch.save(ai.model.state_dict(), os.path.join(MODELS, f'model_{i + 1}.pt'))
    print(f'generation: {i + 1}/{NUM_GENS}')
  stop = time.time()
  print(f'total time: {timedelta(seconds=stop - start)}')

if __name__ == '__main__':
  main()
