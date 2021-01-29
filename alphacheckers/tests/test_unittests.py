from alphacheckers.tests.mock import MockModel, MockTensorflow

# prevent tensorflow from loading
import sys
sys.modules['tensorflow'] = MockTensorflow()


import unittest
import alphacheckers.game.game as game
import numpy as np
from alphacheckers.agent.alphacheckers import Agent, Node

class TestGameFunctions(unittest.TestCase):
    def test_reflect(self):
        jumped = 24
        origin = 28
        to = 21
        can_move = 0x8000
        move = can_move | (jumped << 10) | (origin << 5) | to
        original_move = move
        move = game.reflect_move(move)
        self.assertEqual((move & 0x7C00)  >> 10, 7, "Jumped piece of move reflected incorrectly.")
        self.assertEqual((move & 0x3e0) >> 5, 3, "Origin piece of move reflected incorrectly.")
        self.assertEqual((move & 0x1f), 10, "Destination of piece of move reflected incorrectly.")
        move = game.reflect_move(move)
        self.assertEqual(move, original_move, "Double reflection should equal original move.")

    def test_move_to_index(self):
        state = game.gen_start()
        original = [game.move_to_index(m, 1) for m in game.get_moves(*state)]
        original.sort()
        state[4] = -1
        compare = [game.move_to_index(m, -1) for m in game.get_moves(*state)]
        compare.sort()
        self.assertEqual(original, compare, "Failed to reflect starting moves correctly.")
    


class TestAgentMethods(unittest.TestCase):
    def test_run(self):
        # set up
        mock_nn = MockModel()
        state = game.gen_start()
        agent = Agent(state, mock_nn)

        for i in range(5):
            # get move
            pi, value, action_id, move = agent.run(state)
            # check values
            self.assertGreater(np.sum(pi), 0, "pi vector sum should be greater then 0")
            self.assertAlmostEqual(np.sum(pi), 1, 3, "sum of pi should be about 1")
            self.assertEqual(move in game.get_moves(*state), True, "move returned from run() should be valid.")
            state = game.run_move(*state, move)


    def test_choose_action(self):
        # set up
        mock_nn = MockModel()
        state = game.gen_start()
        agent = Agent(state, mock_nn)

        # test player = -1
        state[4] = -1
        moves = game.get_moves(*state)
        agent.root = Node(state)
        agent.evaluate(agent.root, [])
        for edge in agent.root.edges:
            edge.N = 1
        probs, values = agent.get_action_values()
        pi, value, action_id, move = agent.choose_action(probs, values)
        self.assertEqual(move in game.get_moves(*state), True, "Suggested move ")


    def test_evaluate(self):
        # set up
        mock_nn = MockModel()
        state = game.gen_start()
        agent = Agent(state, mock_nn)

        # test player = -1
        state[4] = -1
        moves = game.get_moves(*state)
        moves.sort()
        agent.root = Node(state)
        agent.evaluate(agent.root, [])
        edge_moves = [edge.move for edge in agent.root.edges]
        edge_moves.sort()
        self.assertEqual(edge_moves, moves, "Evaluate assigns incorrect move to edge.")

    def test_get_preds(self):
        # set up
        mock_nn = MockModel()
        state = game.gen_start()
        state[4] = 1
        valid_moves = game.get_moves(*state)

        agent = Agent(state, mock_nn)
        value, probs, valid = agent.get_preds(state)    
        self.assertGreater(np.sum(probs), 0, "pi vector sum should be greater then 0")
        self.assertAlmostEqual(np.sum(probs), 1, 3, "sum of pi should be about 1")
        # check whether all preds are valid
        for i in range(len(probs)):
            if probs[i] > 0.01:
                # if move is invalid, this should fail  
                action = game.hard_index_to_move(i, valid_moves, state[4])
                self.assertEqual(action in valid_moves, True, "pi vector suggesting invalid moves")



if __name__ == '__main__':
    unittest.main()