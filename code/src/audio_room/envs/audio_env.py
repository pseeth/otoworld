import gym
from gym import error, spaces, utils
from gym.utils import seeding


class AudioEnv(gym.Env):
	def __init__(self):
		print("Environment initialized")

	def step(self, action):
		print("Stepped forward")

	def reset(self):
		print("Environment reset")

	def render(self, close=False):
		print("Rendered")
