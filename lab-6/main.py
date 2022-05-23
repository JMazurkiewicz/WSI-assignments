import gym

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    observation, info = env.reset(seed=0xDEADBEEF, return_info=True)

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation, info = env.reset(return_info=True)

    env.close()
