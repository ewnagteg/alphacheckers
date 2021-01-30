import alphacheckers.game.game as game
from alphacheckers.agent.memory import Memory
import alphacheckers.config.config as config

def run_matches(player1, player2, EPISOPES, memory: Memory = None, iteration=0, log_games = False):
    done = 0
    turn = 0
    scores = { 'wins':0, 'draws':0, 'loses':0 }
    for episode in range(EPISOPES):
        # run mcts
        state = game.gen_start()
        player1.cache = {}
        player2.cache = {}

        player1.reset()
        player2.reset()
        player1.set_state(state)
        player2.set_state(state)
        players = {}
        if player1.player == 1:
            players = { 1: player1, -1: player2 }
        else:
            players = {-1: player1, 1: player2 }

        done = 0
        turn = 0
        print('current episode: {}'.format(episode))
        tau = 1
        while done == 0:
            if log_games:
                game.render(*state)
            turn += 1
            if turn > config.TAU:
                tau = 0
            pi, value, idx, action = players[state[4]].run(state, tau)
            if memory != None:
                memory.commit_stmemory([*state], pi, value)
            # step
            state = game.run_move(*state, action)
            # check if game done
            if len(game.get_moves(*state)) == 0 or turn > 80:
                done = 1
                winner = game.winner(*state)
                if winner == 0:
                    scores['draws'] = scores['draws'] + 1
                elif winner == 1:
                    scores['wins'] = scores['wins'] + 1
                else:
                    scores['loses'] = scores['loses'] + 1
                    
                if memory != None:
                    for move in memory.stmemory:
                        if winner == 0:
                            move['value'] = 0
                        elif move['state'][4] == winner:
                            move['value'] = 1
                        else:
                            move['value'] = -1
                    memory.commit_ltmemory()
    return scores
