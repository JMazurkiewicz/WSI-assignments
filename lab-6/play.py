import random
import sys
from time import sleep
import gym
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('This program expects filename with qtable as the only argument.')
    else:
        qtable = np.loadtxt(sys.argv[1])
        env = gym.make("Taxi-v3")
        state = env.reset(seed=random.getrandbits(64))
        done = False
        while not done:
            env.render()
            action = np.argmax(qtable[state])
            state, _, done, _ = env.step(action)
            sleep(0.5)
        env.render()
