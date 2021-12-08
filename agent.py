from collections import deque
from model import mlp
import random
import numpy as np

class Agent(object):
    # A deep q agent. Q values are approximated using a feed-forward neural network
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = .95 # discount factor, how much the agent cares about future rewards
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.model = mlp(state_size, action_size)

    # Add data to the q table
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # decide what action our agent will take
    def act(self, state):
        # take a random action within our action space if random value is less than our current epsilon
        # as epsilon decays, we will take less and less random actions (using nn to predict which action is best)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # return the best action for some state

    # perform experience replay
    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states = np.array([x[0][0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3][0] for x in batch])
        end = np.array([x[4] for x in batch])

        # Improve performace using a neural network by looking ahead
        # Q(s',a) bellman equation
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        # end state target is the reward itself (no looking ahead)
        target[end] = rewards[end]

        # Q(s, a)
        target_f = self.model.predict(states)
        # Make the agent approximate the current state to future discounted rewards
        target_f[range(batch_size), actions] = target

        # update our nn model 
        self.model.fit(states, target_f, epochs=1, verbose=1)

        # reduce exploration rate everytime we perform an experience replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




