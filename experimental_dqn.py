import numpy as np
import flappy_bird_gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from tensorflow.optimizers import Adam

from rl.agents.dqn import DQNAgent