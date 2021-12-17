from tensorflow.python.ops.gen_array_ops import size
from experience import Experience
import time
import os
import numpy as np
from random import uniform
from constants import *
import flappy_bird_gym
import cv2 as cv
from DQN import DQN
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from agent import Agent
from replay_memory import ReplayMemory
from experience import Experience
import tensorflow as tf

class DeepQLearning:
    def __init__(self, env, gamma, memory_capacity, strategy, batch_size, nr_of_actions, episodes, lr, state_shape, add_conv=True):
        self.env = env
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.episodes = episodes
        self.train_model = DQN(state_shape=state_shape, add_conv=add_conv)
        self.target_model = DQN(state_shape=state_shape,  add_conv=add_conv)
        DeepQLearning.copy_weights(self.target_model, self.train_model)
        self.memory = ReplayMemory(memory_capacity)
        self.strategy = strategy
        self.nr_of_actions = nr_of_actions
        self.frame_count = 6
    

    @staticmethod
    def predict(model, inputs):
        return model(np.atleast_2d(inputs.astype('float32'))) ## .astype('bla bla')

    def train(self):
        if not self.memory.can_provide_sample(self.batch_size):
            return 0

        sample = self.memory.get_sample(self.batch_size)
        states, actions, new_states, rewards, dones = ReplayMemory.unzip_sample(sample)
        value_next = np.max(DeepQLearning.predict(self.target_model, new_states), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                DeepQLearning.predict(self.train_model, states) * tf.one_hot(actions, self.nr_of_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.train_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss


    def get_action(self, states, step):
        if np.random.random() < self.strategy.get_exploration_rate(step):
            return np.random.choice(self.nr_of_actions)
        else:
            return np.argmax(DeepQLearning.predict(self.train_model, states)[0])

    @staticmethod
    def copy_weights(net, net_to_copy):
        variables1 = net.trainable_variables
        variables2 = net_to_copy.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


    def process_obs(self, obs, prev_obs=[(0, 0)]):
        # # from list to np array
        # np_array = np.array(obs)
        # # this array needs to be rotated and flipped horizontally
        # rotated = cv.rotate(np_array, cv.ROTATE_90_CLOCKWISE)
        # flipped = cv.flip(rotated, 1)
        # # then it's resized and grayscaled
        # resized = cv.resize(flipped, (14, 25), interpolation = cv.INTER_AREA)
        # grayscale = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        # processed = np.expand_dims(grayscale, axis=0)
        # processed = np.expand_dims(processed, axis=3)
        # return processed
        # print(obs, prev_obs)
        speed = obs[1] - prev_obs[0][1]
        x, y = obs[0], obs[1]
        new_obs = (x, y, speed) 
        return np.expand_dims(new_obs, axis=0)


    def play_game(self, copy_step):
        iter = 1
        done = False
        # prev_obs = np.zeros((1, 25, 14, 1))
        obs = self.process_obs(self.env.reset())
        losses = list()
        score = 0
        while not done:
            action = self.get_action(obs, iter)
            # prev_prev_obs = prev_obs
            prev_obs = obs
            # cv.imshow('obs', (prev_obs)[0])
            # cv.waitKey(0)
            obs, reward, done, info = env.step(action)
            env.render()
            obs = self.process_obs(obs, prev_obs)
            if done:
                score = info['score']
                reward = -2000
                env.reset()
            else:
                reward = 1
                if info['score'] > score:
                    reward = 100
                
                
                
            
            if self.frame_count % iter == 0:
                
                exp = Experience(prev_obs, action, obs, reward, done)
                self.memory.push(exp)

                loss = self.train()
                if isinstance(loss, int):
                    losses.append(loss)
                else:
                    losses.append(loss.numpy())
                
                
                if (iter // self.frame_count) % copy_step == 0 or score < info['score']:
                    DeepQLearning.copy_weights(self.target_model, self.train_model)
                    if score < info['score']:
                        self.memory.push(exp)

        iter += 1
        return score, np.mean(losses)




    def learn(self, max_steps=None):
        max_score = 0
        for episode in range(self.episodes):
            score, loss = self.play_game(copy_step=100)
            max_score = max(score, max_score)
            print('Episode', episode, 'max score', max_score, 'score', score,'Loss', loss)


if __name__ == '__main__':
    # env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    env = flappy_bird_gym.make("FlappyBird-v0", normalize_obs=False)
    state_shape = (3,) # (25, 14, 1)
    gamma = 1
    memory_capacity = 5000
    strategy = EpsilonGreedyStrategy(0.1, 0, 0.0001)
    batch_size = 32
    nr_of_actions = 2
    lr = 0.001
    episodes = 100000
    deep_q_learning_bot = DeepQLearning(env, gamma, memory_capacity, strategy, batch_size, nr_of_actions, episodes, lr, state_shape=state_shape, add_conv=False)
    deep_q_learning_bot.learn()
    env.close()