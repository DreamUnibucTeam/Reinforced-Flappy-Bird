import os
import json
import sys
import random
import time
from collections import deque

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flappy_bird_gym

# Constants
GAMMA = 0.99
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.01
NUMBER_OF_ITERATIONS = 2000000
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 32

scores = []
episode_number = 0
score = 0
iteration_episodes = []
iteration_scores = []

path = "."

def save_data(timestamp, model, optimizer):
    torch.save(model.state_dict(), f"{path}/model_state_dict/pretrained_model_{timestamp}.pth")
    torch.save(optimizer.state_dict(), f"{path}/optimizer_state_dict/pretrained_model_{timestamp}.pth")
    torch.save(model, f"{path}/pretrained_model/current_model_{timestamp}.pth")
    with open(f"{path}/results/result_{timestamp}.json", "w") as f:
        json.dump({
            'episodes': list(range(0, episode_number)),
            'scores': scores
        }, f)
    with open(f"{path}/results_iterations/result_{timestamp}.json", "w") as f:
        json.dump({
            'timestamps': list(range(0, timestamp)),
            'episodes': iteration_episodes,
            'scores': iteration_scores
        }, f)


def convert_state(image):
    return image_to_tensor(resize_and_bgr2gray(image))


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv.cvtColor(cv.resize(image, (84, 84)), cv.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


class DQN(nn.Module):
    def __init__(self, actions):
        super(DQN, self).__init__()

        self.actions = actions
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.01
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.batch_size = 32

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=actions)
        )

    def forward(self, x):
        return self.network(x)


class DQN_Trainer:
    def __init__(self, env):
        self.env = env
        self.model = DQN(env.action_space.n)

    def learn(self, start=0):
        global score, scores, episode_number, iteration_episodes, iteration_scores
        optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        criterion = nn.MSELoss()

        replay_memory = deque(maxlen=self.model.replay_memory_size)
        obs = self.env.reset()

        obs = convert_state(obs)
        state = torch.cat((obs, obs, obs, obs)).unsqueeze(0)

        epsilon_decrements = np.linspace(self.model.initial_epsilon, self.model.final_epsilon, self.model.number_of_iterations)
        epsilon = epsilon_decrements[start]

        scores = []

        for iteration in range(start, self.model.number_of_iterations):
            # Get the Q-Value for the current state
            q_values = self.model(state)[0]

            # Exploration - Exlpoitation
            random_action = random.random() <= epsilon
            action = random.randint(0, 1) if random_action else torch.argmax(q_values).item()

            # Execute the chosen action
            new_obs, reward, done, info = env.step(action)
            reward = -1000 if done else 0
            new_obs = convert_state(new_obs)
            new_state = torch.cat((state.squeeze(0)[1:, :, :], new_obs)).unsqueeze(0)

            # Check if new episode
            iteration_scores.append(score)
            iteration_episodes.append(episode_number)
            if done:
                scores.append(score)
                score = 0
                episode_number += 1
            else:
                score = info["score"]

            # Add the new state to the replay memory
            action = torch.from_numpy(np.array([int(action == 0), int(action == 1)], dtype=np.float32)).unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
            replay_memory.append((state, action, reward, new_state, done))

            epsilon = epsilon_decrements[iteration]

            # Select a batch of random (at most 32) states and train the DQN with them
            repl = list(replay_memory)
            batch = random.sample(repl, min(len(repl), self.model.batch_size))

            state_batch = torch.cat(tuple(state[0] for state in batch))
            action_batch = torch.cat(tuple(state[1] for state in batch))
            reward_batch = torch.cat(tuple(state[2] for state in batch))
            new_state_batch = torch.cat(tuple(state[3] for state in batch))

            # Execute the learning process and compute the good q-values
            output_batch = self.model(new_state_batch)
            y_batch = torch.cat(tuple(reward_batch[i] if batch[i][4]
                                      else reward_batch[i] + self.model.gamma * torch.max(output_batch[i])
                                      for i in range(len(batch))))

            # q(s, a) = rew + gamma * max(s_new, a==0 sau a==1)         

            q_value = torch.sum(self.model(state_batch) * action_batch, dim=1)
            # (s_new, 0), (s_new, 1)
            optimizer.zero_grad()

            y_batch = y_batch.detach()
            loss = criterion(q_value, y_batch)

            loss.backward()
            optimizer.step()

            scores.append(info["score"])
            # env.render()
            # time.sleep(1 / 250)

            state = new_state
            if done:
                obs = env.reset()
            if iteration != 0 and iteration % 25000 == 0:
                torch.save(self.model, f"pretrained_model_{iteration}.pth")

            if iteration % 500 == 0:
                print(f"Iteration: {iteration}, Q max: {np.max(q_values.detach().numpy())}, Score: {max(scores)}")
                scores = []

    def play(self):
        state = self.env.reset()
        state = convert_state(state)
        state = torch.cat((state, state, state, state)).unsqueeze(0)

        while True:
            output = self.model(state)[0]
            action = torch.argmax(output).item()

            new_state, _, done, _ = self.env.step(action)
            new_state = convert_state(new_state)
            new_state = torch.cat((state.squeeze(0)[1:, :, :], new_state)).unsqueeze(0)

            self.env.render()
            time.sleep(1/30)

            state = new_state
            if done:
                break

if __name__ == "__main__":
    mode = "test"
    if len(sys.argv) == 2:
        if not sys.argv[1] in ["train", "test"]:
            raise Exception("Bad mode specified (should be test or train)")
        mode = sys.argv[1] 
    
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    dqn_bot = DQN_Trainer(env)

    if mode == "train":
        dqn_bot.learn()
    elif mode == "test":
        dqn_bot.model = torch.load("saved/dqn/pretrained_good.pth", map_location="cpu")
        dqn_bot.play()
    