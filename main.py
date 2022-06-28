from game import Game

game = Game()
while game.running:
    game.update()
    game.render()
