# 使用系统python而不是conda的base环境
import gymnasium as gym 
from stable_baselines3 import A2C, PPO
# # Create the environment
env = gym.make('LunarLander-v2',render_mode="human")  # continuous: LunarLanderContinuous-v2

# model = A2C('MlpPolicy', env, verbose=1)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100)

episodes = 10
vec_env = model.get_env()
obs = vec_env.reset()
for ep in range(episodes):
	
	done = False
	while not done:
		action, _states = model.predict(obs)
		obs, rewards, done, info = vec_env.step(action)
		env.render()
		print(rewards)

env.close()

# for step in range(200):
# 	env.render()
# 	# take random action
# 	# env.step(env.action_space.sample())
# 	observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
# 	print(reward, terminated)

# # sample action:
# print("sample action:", env.action_space.sample())

# # observation space shape:
# print("observation space shape:", env.observation_space.shape)

# # sample observation:
# print("sample observation:", env.observation_space.sample())



# import gymnasium as gym 


# env = gym.make('LunarLander-v2',render_mode="human")  # continuous: LunarLanderContinuous-v2
# env.reset()

# for step in range(200):
# 	env.render()
# 	# take random action
# 	env.step(env.action_space.sample())

# env.close()