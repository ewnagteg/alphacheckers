import alphacheckers.game.game as game
import numpy as np

class MockModel():
    def __init__(self):
        super().__init__()
        self.state = game.gen_start()

    def predict(self, state_input):
        """
        Will generate mock Model output where value = 0 and each move is equally probable
        """
        moves = game.get_moves(*self.state)
        
        pi = np.zeros((256,))
        for m in moves:
            pi[game.move_to_index(m, self.state[4])] = 1.
        pi /= len(moves)

        value = 0
        a1 = np.array([pi])
        a0 = np.zeros((1,1))
        return [a0, a1]

    def convert_to_model_input(self, state):
        self.state = state
        return None
        
class MockTensorflow():
    def __init__(self):
        pass
