import time
import os
import numpy as np
from random import uniform
from constants import *
import flappy_bird_gym


class QLearning:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((X_DISTANCE_MAX, Y_DISTANCE_MAX, env.action_space.n))
        self.block_size = BLOCK
        self.episodes = NUM_EPISODES
        self.learning_rate = LEARNING_RATE
        self.discount_rate = DISCOUNT_RATE
        self.rewards_all_episodes = []
        self.scores_all_episodes = []
        self.results_path = "saved/q_learning"

    def compute_state(self, obs):
        return obs[0] // self.block_size, (obs[1] + OFFSET) // 100

    def update_q_table(self, obs, new_obs, action, reward):
        # obs_x, obs_y = min(obs[0], 280) // BLOCK, (obs[1] + OFFSET) // BLOCK
        # new_obs_x, new_obs_y = min(new_obs[0], 280) // BLOCK, (new_obs[1] + OFFSET) // BLOCK
        obs_x, obs_y = self.compute_state(obs)
        new_obs_x, new_obs_y = self.compute_state(new_obs)
        self.q_table[obs_x, obs_y, action] = self.q_table[obs_x, obs_y, action] * (1 - self.learning_rate) + self.learning_rate * (
                reward + self.discount_rate * np.max(self.q_table[new_obs_x, new_obs_y, :]))

    def load_q_table(self, path):
        if not os.path.exists(path):
            raise Exception(f"There is not file located at {path} to load")

        with open(path, 'rb') as file:
            self.q_table = np.load(file)

    def save_q_table(self, path, details="best"):
        filename = f"{self.results_path}/{path}_{details}.npy"
        with open(filename, 'wb') as file:
            np.save(file, self.q_table)

    def learn(self, max_steps=None):
        best_score = 0

        for episode in range(self.episodes):
            obs = self.env.reset()
            score = 0
            always_on = max_steps is None
            current_step = 1
            done = False
            rewards_current_episode = 0

            while always_on or current_step <= max_steps:
                current_step += 1
                # Exploration - exploitation
                # exploration_rate_thresh = uniform(0, 1)
                # if exploration_rate_thresh > exploration_rate:
                #     obs_x, obs_y = int(min(obs[0], 280)) // BLOCK, (int(obs[1]) + OFFSET) // BLOCK
                #     action = np.argmax(q_table[obs_x, obs_y, :])
                # else:
                #     obs_x, obs_y = int(min(obs[0], 280)) // BLOCK, (int(obs[1]) + OFFSET) // BLOCK
                #     action = np.argmax(q_table[obs_x, obs_y, :])
                    # action = env.action_space.sample()
                obs_x, obs_y = self.compute_state(obs.astype(int))
                action = np.argmax(self.q_table[obs_x, obs_y, :])

                new_obs, reward, done, info = self.env.step(action)
                reward = -1000 if done else 0
                # reward += 0.1 if score != info["score"] else 0

                score = info["score"]
                if score > 10000:
                    print(f"Episode {episode}: Score {score}")

                self.update_q_table(obs.astype(int), new_obs.astype(int), action, reward)
                obs = new_obs
                rewards_current_episode += reward

                # if episode > 0:
                #     env.render()
                #     time.sleep(1 / 20)

                if done:
                    break

            if score >= 1000 and score >= best_score:
                best_score = score
                self.save_q_table(f"result_episode_{episode}", f"score_{score}")

            # Decay the exploration rate
            # exploration_rate = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(
            #     -EXPLORATION_RATE_DECAY * episode)
            self.rewards_all_episodes.append(rewards_current_episode)
            self.scores_all_episodes.append(score)
            if episode % 1000 == 0:
                print(
                    f"Episode {episode}: Reward: {rewards_current_episode if episode == 0 else np.max(np.array(self.rewards_all_episodes)[episode - 1000:episode])}, Score {score if episode == 0 else np.max(np.array(self.scores_all_episodes)[episode - 1000:episode])}")


if __name__ == "__main__":
    env = flappy_bird_gym.make("FlappyBird-v0", normalize_obs=False)
    q_learning_bot = QLearning(env)
    q_learning_bot.learn()

