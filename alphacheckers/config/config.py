ALPHA = 0.8 # scale for dirichlet noise, paper used 10 / branching factor
			   # see https://en.wikipedia.org/wiki/Game_complexity

TAU = 30 # number of turns at which tau == 1, after which it is set to zero

# Self play constants
EPISODES = 100

x = 0.6 # epsilon, paper uses 0.75, some other people used 0.5
MCTS_SIMS = 30
MEMORY_SIZE = 30000
CPUCT = 1

HIDDEN_LAYERS = 10

# Training constants
BATCH_SIZE = 256
EPOCHS = 2
REG_CONST = 0.0001
LEARNING_RATE = 0.001
MOMENTUM = 0.9
TRAINING_LOOPS = 20
TRAINIG_CONSANT = 1

# Directory containing models
# Where tf saves the training models
MODELS_DIR = '/content/drive/MyDrive/'