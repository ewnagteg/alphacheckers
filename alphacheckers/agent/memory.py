import numpy as np
from collections import deque
import alphacheckers.config.config as config

class Memory:
    def __init__(self, MEMORY_SIZE):
        self.MEMORY_SIZE = config.MEMORY_SIZE
        self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)

    def commit_stmemory(self, state, pi, value):
        self.stmemory.append({
                'state': state,
                'pi': pi,
                'value': value
            })

    def commit_ltmemory(self):
        print('committing ltmemory')
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()
        print('ltmemory size now: {}'.format(len(self.ltmemory)))

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)