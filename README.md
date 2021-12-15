# Reinforced Flappy Bird

## Installation:
1. Use the project in PyCharm, with the provided environment in the `venv` folder. In no `venv` is provided, just run in terminal:
```bash
python -m venv venv
```
2. Activate the `venv` in terminal:
```bash
source /venv/Scripts/activate # for windows
source /venv/bin/activate # for unix
```
3. If created the `venv`, run:
```bash
pip install flappy-bird-env
pip install tensorflow opencv-python
pip install matplotlib
```

4. If created the `venv`, add the contents of `modified_env` folder into the `venv/Lib/site-packages`
5. To run the project:
```bash
python main.py # for the test run of the project
python q_learning.py # for the q_learning algorithm
```

## Modifications to the ENV:
1. Add normalize_obs=False parameter when creating a simple env
2. Make the background black when training a DQN: FILL_BACKGROUND_COLOR in renderer => change to black (0,0,0)
3. Remove bird rotation: in renderer.py, add to display_surface parameter rotate_bird: bool = False and add an if statement
4. Add less randomization to new generated pipes (add start_y = [20, 30, 40, …, 90] and select a random one of them)
5. Modify the PIPE_HEIGHT in SimpleEnv and in RGBEnv
6. Maybe remove the score mark (in FlappyBirdEnvRGB.render, make show_score=False)
7. Retrieve velocity from env (add with_velocity: bool = True to simple env's get observation and reset and step)

