import tensorflow as tf
import alphacheckers.config.config as config
import numpy as np
import alphacheckers.game.game as game


class Res_CNN():
    def __init__(self, reg_const=config.REG_CONST, learning_rate=config.LEARNING_RATE, input_dim=(1,8,8,4), hidden_layers=[5]):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.output_dim = 32*8
        self.model = self.build()
        self._map = {
            0: 2,
            1: 3,
            2: 0,
            3: 1
        }

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = tf.keras.layers.Conv2D(
        filters = filters
        , kernel_size = kernel_size
        , padding = 'same'
        , use_bias=False
        , activation='linear'
        , kernel_regularizer = tf.keras.regularizers.l2(self.reg_const)
        
        )(x)

        x = tf.keras.layers.BatchNormalization(axis=3)(x)

        x = tf.keras.layers.add([input_block, x])

        x = tf.keras.layers.LeakyReLU()(x)

        return (x)

    def conv_layer(self, x, filters, kernel_size):

        x = tf.keras.layers.Conv2D(
        filters = filters
        , kernel_size = kernel_size
        , padding = 'same'
        , use_bias=False
        , activation='linear'
        , kernel_regularizer = tf.keras.regularizers.l2(self.reg_const)
        
        )(x)

        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.LeakyReLU()(x)

        return (x)

    def value_head(self, x):
        x = tf.keras.layers.Conv2D(
        filters = 1
        , kernel_size = (1,1)
        , padding = 'same'
        , use_bias=False
        , activation='linear'
        , kernel_regularizer = tf.keras.regularizers.l2(self.reg_const)
        )(x)


        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(180, use_bias=False, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(self.reg_const))(x)

        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Dense(1, use_bias=False, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(self.reg_const), name = 'value_head')(x)

        return (x)

    def policy_head(self, x):
        """
        The output of this is a (256) vector of floats.
        Use alphacheckers.game.game.move_to_index(move, player) to get a index for a given move.
        This always assumes that the zero square is in the bottom right.
        """
        x = tf.keras.layers.Conv2D(
        filters = 2
        , kernel_size = (1,1)
        , padding = 'same'
        , use_bias=False
        , activation='linear'
        , kernel_regularizer = tf.keras.regularizers.l2(self.reg_const)
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32*8, 
                                activation='softmax', 
                                use_bias=False, 
                                kernel_regularizer=tf.keras.regularizers.l2(self.reg_const),
                                name='policy_head')(x)
        return (x)
        
    def build(self):
        # HWC - cpu doesn't support CHW
        main_input = tf.keras.layers.Input(shape = (8,8,4), name = 'main_input')
        x = self.conv_layer(main_input, 128, (3,3))
        for h in range(10):
            x = self.residual_layer(x, 128, (3,3))

        vh = self.value_head(x)
        ph = self.policy_head(x)
        model = tf.keras.Model(inputs=[main_input], outputs=[vh, ph])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
            optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate, momentum = config.MOMENTUM),    
            loss_weights={'value_head': 0.5, 'policy_head': 0.5}    
            )
        return model
        
    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split = validation_split, batch_size = batch_size)

    def convert_to_model_input(self, state):
        """
        Parameters:
        state: current board state
        Returns:
            Model input, which is always from perspective of current player.
        """
        board = np.zeros((1, 8, 8, 4))
        check = state[0]|state[1]|state[2]|state[3]
        for i in range(32):
            tn = 1 << i
            x = game.getX(i)
            y = game.getY(i)
            if state[4] == -1:
                x = 7 - x
                y = 7 - y
            for idx in range(0,4):
                index = idx
                if state[4] == -1:
                    index = self._map[index]
                if (tn&check) != 0:
                    if (tn & state[idx]) == tn:
                        board[0, x, y, index] = 1 if index == 0 or index == 1 else -1
        return board
    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)

    def compile_model(self):
        self.model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
            optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate, momentum = config.MOMENTUM),    
            loss_weights={'value_head': 0.5, 'policy_head': 0.5}    
            )
            
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

def softmax_cross_entropy_with_logits(y_true, y_pred):

    p = y_pred
    pi = y_true

    zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0) 
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

    return loss