from game import Game

game = Game()
game.ai.train()
game.ai.test()
while game.running:
  game.update()
  game.render()
