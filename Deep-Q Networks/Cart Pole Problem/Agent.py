import gymnasium as gym
import torch
import random
from collections import deque
from Model import Linear_QNetwork, QTrainer
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
INITIAL_EPSILON = 0.5
MIN_EPSILON = 0.01
DECAY_RATE = 0.99
LR = 0.001

class Agent:
    def __init__(self):
        self.num_of_games = 0
        self.gamma = 0.9
        self.epsilon = INITIAL_EPSILON
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNetwork(4, 256, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    def decay_epsilon(self):
        #self.epsilon = MIN_EPSILON + (INITIAL_EPSILON - MIN_EPSILON) * math.exp(-DECAY_RATE * self.num_of_games)
        self.epsilon *= DECAY_RATE
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        # Random actions: tradeoff exploration / exploitation
        move = 0
        if random.random() < self.epsilon:
            move = random.randint(0, 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            
        return move

def train():
    agent = Agent()
    env = gym.make('CartPole-v1', render_mode='human')
    last_100 = deque(maxlen=100)
    best_score = 0
    while True:
        observation = env.reset()
        observation = observation[0]
        agent.decay_epsilon()
        for t in range(1000):
            env.render()
            action = agent.get_action(observation)
            
            next_observation, reward, done, _, _ = env.step(action)
            agent.train_short_memory(observation, action, reward, next_observation, done)
            
            agent.remember(observation, action, reward, next_observation, done)
            
            if done:
                last_100.append(t+1)
                if t+1 > best_score:
                    best_score = t+1
                print("Game {} over after {} timesteps. Best score: {} Average score of last 100: {} Epsilon: {}"
                      .format(agent.num_of_games, t+1, best_score, round(sum(last_100)/len(last_100), 2), agent.epsilon))
                agent.num_of_games += 1
                agent.train_long_memory()
                break
        
if __name__ == '__main__':
    train()