import time
import flappy_bird_gym
env = flappy_bird_gym.make("FlappyBird-rgb-v0")


obs = env.reset()
round = 0
while True:
    # Next action:
    # (feed the observation to your agent here)
    # action = env.action_space.sample()
    action = int(round % 15 == 0)
    round += 1


    # Processing:
    obs, reward, done, info = env.step(action)
    print(action, obs.shape, reward, done, info)

    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 30)  # FPS
    # time.sleep(1/30)

    # Checking if the player is still alive
    if done:
        break

env.close()
