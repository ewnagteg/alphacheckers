import alphacheckers.config.config as config
import alphacheckers.game.game as game
import alphacheckers.agent.model as model
import numpy as np
import pickle
import random

class Edge():
    def __init__(self, parent, child, prior, move, pid, player_turn):
        self.parent = parent
        self.child = child
        self.move = move
        # index of move in p vector, see game.move_to_index for details
        self.player_turn = player_turn
        self.pid = pid
        self.P = prior
        self.W = 0
        self.Q = 0
        self.N = 0

class Node():
    def __init__(self, state):
        self.state = [*state]
        self.edges = []
    def is_leaf(self):
        return len(self.edges) == 0

class Agent(object):
    def __init__(self, state, nn):
        self.root = None
        self.cpuct = config.CPUCT
        self.simulations = config.MCTS_SIMS
        self.model = nn
        self.player = 1
        self.root = Node(state)
        self.action_size = 32*8
        self.cache = {}

    def select(self):
        current = self.root
        # stack of edges, makes backprop easier
        stack = []
        x = config.x
        while not current.is_leaf():
            max_QU = -float('Inf')
            alpha = 0
            if current == self.root:
                x = config.x
                alpha = np.random.dirichlet([config.ALPHA] * len(current.edges))
            else:
                x = 0
                alpha = [0] * len(current.edges)

            Nb = 0
            for edge in current.edges:
                Nb += edge.N
            best_edge = current.edges[0]
            for idx, edge in enumerate(current.edges):
                U = self.cpuct * (config.x * edge.P + (1-config.x) * alpha[idx]) * np.sqrt(Nb) / (1 + edge.N)
                Q = edge.Q
                if U + Q > max_QU:
                    max_QU = Q + U
                    best_edge = edge
            current = best_edge.child
            stack.append(best_edge)
        return current, stack

    def backpropagate(self, leaf, value, stack):
        player_leaf = leaf.state[4]
        while len(stack) > 0:
            edge = stack.pop()
            # value is valuation from current player's perspective, therefore if edge turn is diffrent
            # then we should flip value sign.
            # This makes sense even when edge goes from, for example, player 1 to player 1, which is possible
            # in this implenatation.
            direction = 1
            if edge.player_turn != player_leaf:
                direction = -1
            edge.N += 1
            edge.W += value * direction
            edge.Q = edge.W / edge.N


    def evaluate(self, node: Node, stack):
        value, probs, valid = self.get_preds(node.state)
        probs = probs[valid]
        mlist = game.get_moves(*node.state)
        if len(mlist) == 0:
            return value[0]
        for idx, move_index in enumerate(valid):
            action = game.hard_index_to_move(move_index, mlist, node.state[4])
            new_state = game.run_move(*node.state, action)
            new_node = Node(new_state)
            edge = Edge(node, new_node, probs[idx], action, move_index, node.state[4])
            node.edges.append(edge)
        return value[0]

    def get_preds(self, state):
        # check cache first
        key = game.get_state_key(*state)
        output = None
        if key in self.cache:
            output = self.cache[key]
        else:
            state_input = self.model.convert_to_model_input(state)
            output = self.model.predict(state_input)
            self.cache[key] = output

        value = output[0][0]
        logits = output[1][0]

        mask, valid = game.get_moves_mask(state, state[4])
        mask = np.reshape(mask, logits.shape)
        mask = mask.reshape((256))
        logits[mask] = -100
        odds = np.exp(logits)
        probs = odds / np.sum(odds)
        return value, probs, valid

    def get_action_values(self):
        pi =  np.zeros(self.action_size, dtype=np.int32)
        values =  np.zeros(self.action_size, dtype=np.float32)
        for edge in self.root.edges:
            i = game.move_to_index(edge.move, edge.player_turn)

            pi[i] = edge.N
            values[i] = edge.Q
        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def choose_action(self, pi, values, tau=1):
        action_idx = None
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action_idx = random.choice(actions)[0]
        else:
            actions = np.random.multinomial(1, pi)
            action_idx = np.where(actions==1)[0][0]

        # needs to consider reflections for player = -1
        # choose action without tau
        act = game.index_to_move(action_idx, self.root.state[4])
        for i, edge in enumerate(self.root.edges):
            if (edge.move&0x3FF) == act:
                return pi, values[action_idx], action_idx, edge.move
        game.render(*self.root.state)
        game.pretty_print_move(act)
        raise Exception("could not find move")

    def run(self, state, tau=0.5):
        self.root = Node(state)
        for i in range(self.simulations):
            self.simulate()
        pi, values = self.get_action_values()
        return self.choose_action(pi, values, tau)

    def simulate(self):
        leaf, stack = self.select()
        value = self.evaluate(leaf, stack)
        self.backpropagate(leaf, value, stack)

    def replay(self, ltmemory, iteration=0):
        loops = round((len(ltmemory) / config.BATCH_SIZE) * config.TRAINIG_CONSANT)

        # log memory for debugging
        log = open(config.MODELS_DIR + "ltmemory_dump_{}.pkl".format(iteration), "wb")
        pickle.dump(ltmemory, log)
        log.close()
        print('saved ltmemory')
        # to load
        # log = open("ltmemory_dump_0.pkl", "rb") 
        # ltmemory = pickle.load(log)
        for i in range(loops):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))
            training_states = np.array([self.model.convert_to_model_input(row['state'])[0] for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])
                                , 'policy_head': np.array([row['pi'] for row in minibatch])} 
            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size = config.BATCH_SIZE)
            print('NEW LOSS %s', fit.history)

    def reset(self):
        self.root = None

    def set_state(self, state):
        self.root = Node(state)


    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
    