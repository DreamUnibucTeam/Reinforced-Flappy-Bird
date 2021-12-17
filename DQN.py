import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np

class DQN(Model):
    def __init__(self, state_shape=(25, 14, 1), nr_of_actions=2, add_conv=True):
        super(DQN, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=state_shape)
        self.hidden_layers = []
        if add_conv:
            self.hidden_layers.append(layers.Conv2D(16, (4, 4), activation='relu', kernel_initializer='RandomNormal'))
        self.hidden_layers.append(layers.Flatten())
        self.hidden_layers.append(layers.Dense(32, activation='relu'))
        self.hidden_layers.append(layers.Dense(16, activation='relu'))
        self.hidden_layers.append(layers.Dense(8, activation='relu'))
        self.hidden_layers.append(layers.Dense(4, activation='relu'))
        self.hidden_layers.append(layers.Dense(4, activation='relu'))
        self.output_layer = layers.Dense(nr_of_actions, activation='tanh', kernel_initializer='RandomNormal')


    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output



if __name__ == '__main__':
    model = DQN()
    model.summary()