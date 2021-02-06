import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model", type=int, help="model number to load")
parser.add_argument("-s", "--switch", action="store_true", help="switch sides of human, ai")
args = parser.parse_args()

import alphacheckers.game.game as game
import alphacheckers.agent.alphacheckers as agent
import alphacheckers.agent.model as model
import alphacheckers.config.config as config

class PlayerAgent():
    def __init__(self, player):
        self.player = player
    def run(self, state):
        moves = game.get_moves(*state)
        while 1:
            try:
                input_move = input("Enter move: ")
                move_dict = {}
                for m in game.get_moves(*state):
                    move_dict["{}-{}".format( (m&0x3e0) >> 5, m&0x1f)] = m
                return [], 0, 0, move_dict[input_move]
            except Exception as e:
                print("invalid move")
                print(e)
class Game():
    def __init__(self, model_num):
        self.state = game.gen_start()
        self.nn = model.Res_CNN()
        self.nn.load_model(config.MODELS_DIR + 'model-{}.ckpt'.format(model_num))
        self.nn.compile_model()
        self.player = PlayerAgent(1)
        self.ai = agent.Agent(self.state, self.nn)
        if args.switch:
            self.players = { 1: self.player, -1: self.ai }
        else:
            self.players = { -1: self.player, 1: self.ai }

    def play(self):
        while len(game.get_moves(*self.state)) > 0:
            game.render(*self.state)
            print("".join(["{}-{}    ".format( (m&0x3e0) >> 5, m&0x1f) for m in game.get_moves(*self.state)]))
            _, _, _, move = self.players[self.state[4]].run(self.state)
            self.state = game.run_move(*self.state, move)
        print("Winner: {}".format(self.state[4] * -1))

def main():
    g = Game(args.model)
    g.play()

if __name__ == "__main__":
    main()