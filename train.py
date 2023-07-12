from ai import AI
from game import ConnectFour
import time
from datetime import timedelta

def main():
  start = time.time()
  cf = ConnectFour()
  ai = AI(cf)
  ai.learn()
  stop = time.time()
  print(f'total time: {timedelta(seconds=stop - start)}')

if __name__ == '__main__':
  main()
