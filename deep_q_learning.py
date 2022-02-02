import time
import os
import numpy as np
from random import uniform
from constants import *
import flappy_bird_gym
import cv2 as cv

class DeepQLearning:
    def __init__(self, env):
        self.env = env
        self.episodes = NUM_EPISODES


    def process_obs(self, obs):
        # from list to np array
        np_array = np.array(obs)
        # this array needs to be rotated and flipped horizontally
        rotated = cv.rotate(np_array, cv.ROTATE_90_CLOCKWISE)
        flipped = cv.flip(rotated, 1)
        # then it's resized and grayscaled
        resized = cv.resize(flipped, (28, 51), interpolation = cv.INTER_AREA)
        grayscale = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

        return grayscale

    def learn(self, max_steps=None):

        for episode in range(self.episodes):
            obs = self.env.reset()
            score = 0
            current_step = 1
            done = False
            
            while max_steps is None or current_step <= max_steps:
                img = self.process_obs(obs)
                # random action
                action = env.action_space.sample()

                new_obs, reward, done, info = self.env.step(action)
                score = info['score']
                if done:
                    print(f'Episode {episode}: Score {score}')
                    break

                obs = new_obs


if __name__ == '__main__':
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    deep_q_learning_bot = DeepQLearning(env)
    deep_q_learning_bot.learn()
    env.close()