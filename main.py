from game import Game

game = Game()
game.ai.train()
while game.running:
  game.update()
  game.render()
