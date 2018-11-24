import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.models import load_model
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size, model_dir=None):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01

        if model_dir:
            # loading stored model archtitecture and model weights
            self.load_model(model_dir)
        else:
            # creating model from scratch
            self.model = self._build_model()

    def _build_model(self):

        seqmodel = Sequential()
        seqmodel.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=(80, 80, 1)))
        seqmodel.add(Activation('relu'))
        seqmodel.add(Conv2D(64, (4, 4), strides=(2, 2)))
        seqmodel.add(Activation('relu'))
        seqmodel.add(Conv2D(64, (3, 3), strides=(1, 1)))
        seqmodel.add(Activation('relu'))
        seqmodel.add(Flatten())
        seqmodel.add(Dense(256))
        seqmodel.add(Activation('relu'))
        seqmodel.add(Dense(3))

        adam = Adam(lr=1e-6)
        seqmodel.compile(loss='mse', optimizer=adam)

        return seqmodel

    def remember(self, state, action, reward, next_state, done):

        # store S-A-R-S in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        action = np.argmax(act_values[0])

        return action

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # do the learning
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    # def save_model(self):
    #
    #     # serialize model to HDF5
    #     self.model.save_weights("model-weights.h5")
    #     print("Saved model weights to disk")
    #
    # def load_model(self, model_dir):
    #
    #     # load weights into new model
    #     self.model = self._build_model()
    #     self.model.load_weights(model_dir)
    #     print("Loaded model from disk")
