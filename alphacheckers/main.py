import alphacheckers.config.config as config
import alphacheckers.agent.model as model
import alphacheckers.game.game as game
from alphacheckers.agent.memory import Memory
from alphacheckers.agent.alphacheckers import Agent, Node, Edge
from alphacheckers.agent.funcs import run_matches
import random

# Used for testing agent
class RandomAgent:
    def __init__(self):
        self.player = -1
        self.cache = {}
    def run(self, state, tau):
        moves = game.get_moves(*state)
        return [], [0], 0, random.choice(moves)

    def reset(self):
        return

    def set_state(self, state):
        return

def self_play(player1, player2, iteration, memory=None, log_result=False, save_model=False):
    """
    Parameters:
    player1: Agent used for self play and training
             You can switch which side/player player1 is by setting player field
    player2: other agent used for self play, can be RandomAgent
    iteration (int): Iteration number, if > 0 will load iteration - 1 model
    memory: Used to store targets for training that are generated from self-play, set to None
            If you just want to test a model
    log_result (boolean): Will log result of games.
    """
    if iteration > 0:
        print('loading model...')
        player1.load_model(config.MODELS_DIR + 'model-{}.ckpt'.format(iteration))
        player1.compile_model()
    result = run_matches(player1, player2, config.EPISODES, memory=memory, iteration=iteration)
    if save_model:
        player1.model.model.save(config.MODELS_DIR + 'model-{}.ckpt'.format(iteration))
    if log_result:
        print(result)
    if memory is not None:
        memory.clear_stmemory()
        player1.replay(memory.ltmemory)

def main():
    """
    Trains one iteration of a new model
    """
    memory = Memory(config.MEMORY_SIZE)
    nn = model.Res_CNN()
    agent = Agent(game.gen_start(), nn)
    self_play(agent, agent, 0, memory=memory, save_model=True)