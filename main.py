import gymnasium as gym
from stable_baselines3 import A2C, PPO
env = gym.make('LunarLander-v2', render_mode='human')

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)

episodes = 10
vec_env = model.get_env()
obs = vec_env.reset()

for episode in range(1, episodes+1):
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        print(reward)
env.close()