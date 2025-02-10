#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
from itertools import count
from typing import List
import os
import sys
import math
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN

"""
You can import any package and define any extra function as you need
"""

wandb.init(project='RL3', name='Best_Continued')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

CONSTANT = 10000

GAMMA = 0.99

BATCH_SIZE = 1024

BUFFER_SIZE = 50000

# Epsilon parameters for the decay
# The initial exploration rate, set to 1 (meaning 100% exploration).
EPSILON = 1
# The final exploration rate, set to 0.025.
EPSILON_END = 0.025
# The number of steps after which epsilon starts decaying.
DECAY_EPSILON_AFTER = 5000

# Updating the model params
# The frequency (in steps) at which the target Q-network is updated.
TARGET_UPDATE_FREQUENCY = 5000
# The frequency (in steps) at which the model is saved.
SAVE_MODEL_AFTER = 5000

# Learning rate for the optimizer used during training.
LEARNING_RATE = 1e-4

# Transition is defined with fields state, action, reward, and next_state.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# Initialize a deque named reward_buffer with a single element 0.0 and a maximum length of 100.
reward_buffer = deque([0.0], maxlen=100)

class Agent_DQN(Agent):  # Renamed to DDQN agent
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            parameters for neural network  
            initialize Q net and target Q net
            parameters for replay buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.batch_size = BATCH_SIZE
        self.env = env
        self.action_count = self.env.action_space.n

        in_channels = 4  # For 4 stacked frames as input

        # Main Q-network
        self.Q = DQN(in_channels, self.action_count).to(device)
        # Target Q-network
        self.Q_cap = DQN(in_channels, self.action_count).to(device)
        self.Q_cap.load_state_dict(self.Q.state_dict())

        self.optimizer = optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)

        self.buffer = deque([], maxlen=BUFFER_SIZE)
        self.training_step = 0

        if args.test_dqn:
            print('loading trained model')
            test = torch.load('final_dqn_model_reward_14.5.pth')
            self.Q.load_state_dict(test)
        elif args.train_dqn_again:
            print('Loading model to continue training')
            again = torch.load('final_dqn_model_reward_14.5.pth')
            self.Q.load_state_dict(again)
            self.Q_cap.load_state_dict(self.Q.state_dict())
            # Set epsilon to near the ending value to leverage the pretrained policy

    def init_game_setting(self):
        """
        Testing function will call this function at the beginning of a new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
                
        ###########################
        pass
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        state = np.asarray(observation, dtype=np.float32) / 255
        state = state.transpose(2, 0, 1)
        state = torch.from_numpy(state).unsqueeze(0).to(device)

        with torch.no_grad():
            Q_new = self.Q(state)
        action_idx = torch.argmax(Q_new, dim=1).item()
        return action_idx
    
    def push(self, *args) -> None:
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer.append(Transition(*args))
        ###########################
        
        
    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Initialize an empty list called samples to store the sampled transitions.
        samples = []

        # This loop iterates batch_size times to sample that many transitions from the replay buffer.
        for i in range(batch_size):

            # This index is used to randomly select a transition for sampling.
            idx = random.randrange(len(self.buffer))
            # Appends the selected transition at index idx to the samples list.
            samples.append(self.buffer[idx])
            # Removes the selected transition from the replay buffer to avoid re-sampling the same transition.
            del self.buffer[idx]

        return samples
        ###########################
        # return 
        
    
    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Initialize the episode counter.
        episode = 0

        # Initialize a variable to track the previous mean episode reward.
        exp_reward = 0

        # The training loop continues as long as the mean reward in the reward_buffer is less than 70.
        while np.mean(reward_buffer) < 70:
            # Print the current episode number.
            print("Doing Episode", episode)
            # Initialize a timestamp counter for the current episode.
            t_stamp = 0
            # Initialize the cumulative reward for the current episode.
            epi_reward = 0
            # Reset the environment to start a new episode and get the initial state.
            curr_state = self.env.reset()

            # Enter the inner loop, which runs until the episode is done.
            while True:
                # Check if the episode number is greater than a threshold for epsilon decay threshold
                # if episode > DECAY_EPSILON_AFTER:
                #     # Decay the exploration parameter (epsilon) over time after a certain episode.
                #     epsilon = max(EPSILON_END, epsilon - (epsilon - EPSILON_END) / CONSTANT)
                # else:
                #     # If the episode number is not greater than the threshold, use the initial epsilon value.
                #     epsilon = EPSILON
                epsilon = EPSILON_END
                # Choose an action either by exploiting the current Q-values (with probability 1 - epsilon) or
                # exploring randomly (with probability epsilon).
                if random.random() > epsilon:
                    # If exploiting, use the make_action method to get the action from the trained Q-network.
                    action = self.make_action(curr_state)
                else:
                    # If exploring, choose a random action.
                    action = np.random.randint(0, self.action_count)

                # Take a step in the environment based on the chosen action and obtain the next state, reward, and whether the episode is done.
                next_state, reward, done, _, _ = self.env.step(action)

                # Convert the reward and action to PyTorch tensors.
                tensor_reward = torch.tensor([reward], device=device, dtype=torch.float32)
                tensor_action = torch.tensor([action], dtype=torch.int64, device=device)

                # Preprocess the current states:
                state = np.asarray(curr_state, dtype=np.float32) / 255
                state = state.transpose(2, 0, 1)
                store_buffer_curr_state = torch.from_numpy(state).unsqueeze(0).to(device)

                # Preprocess the next states:
                state = np.asarray(next_state, dtype=np.float32) / 255
                state = state.transpose(2, 0, 1)
                store_buffer_next_state = torch.from_numpy(state).unsqueeze(0).to(device)

                # Store the transition (current state, action, reward, next state) in the replay buffer using the push method.
                self.push(store_buffer_curr_state, tensor_action, tensor_reward, store_buffer_next_state)

                # Update the current state for the next iteration, and accumulate the episode reward.
                curr_state = next_state
                epi_reward += reward

                # If the replay buffer size reaches a certain threshold, optimize the Q-network using the optimize method.
                if len(self.buffer) >= 5000:
                    self.optimize()

                # If the episode is done, append the episode reward to the reward_buffer,
                # increment the episode counter, and exit the inner loop.
                if done:
                    reward_buffer.append(epi_reward)
                    episode += 1
                    break
                # Increment the timestamp counter.
                t_stamp += 1

            # Log metrics every 100 episodes using Weights & Biases (wandb)
            if episode % 100 == 0:
                wandb.log({
                    "reward": np.mean(reward_buffer),
                    "episode": episode,
                    "epsilon": epsilon,
                    "timestamp": t_stamp
                })

            # Update the target Q-network (self.Q_cap) at intervals specified by TARGET_UPDATE_FREQUENCY.
            if episode % TARGET_UPDATE_FREQUENCY == 0:
                self.Q_cap.load_state_dict(self.Q.state_dict())

            # Save the Q-network model at intervals specified by SAVE_MODEL_AFTER.
            if episode % SAVE_MODEL_AFTER == 0:
                torch.save(self.Q.state_dict(), "final_dqn_model_reward_14.5_again.pth")
                print(f"saving model at reward {np.mean(reward_buffer)}")
                exp_reward = np.mean(reward_buffer)
                wandb.log({"exp_reward": exp_reward})

        # Save the final Q-network model and print "Done Wooooo" when the training loop completes.
        torch.save(self.Q.state_dict(), "final_dqn_model_reward_14_again.pth")
        print("Done Wooooo")       
        ###########################

    def optimize(self):
        """
        Perform optimization step for Double DQN.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Ensure there are enough samples in the buffer
        if len(self.buffer) < BATCH_SIZE:
            return

        # Sample a batch of transitions from the replay buffer
        transitions = self.replay_buffer(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Create batches for states, actions, rewards, and next_states
        state_batch = torch.cat(batch.state).to(device)          # Shape: [BATCH_SIZE, 4, 84, 84]
        action_batch = torch.cat(batch.action).unsqueeze(1).to(device)  # Shape: [BATCH_SIZE, 1]
        reward_batch = torch.cat(batch.reward).to(device)        # Shape: [BATCH_SIZE]
        next_state_batch = torch.cat(batch.next_state).to(device)    # Shape: [BATCH_SIZE, 4, 84, 84]

        # Get current Q-values from the Q-network for the chosen actions
        current_q_values = self.Q(state_batch).gather(1, action_batch).squeeze(1)  # Shape: [BATCH_SIZE]

        # Select the best action using the main Q-network (Double DQN)
        next_actions = self.Q(next_state_batch).max(1)[1].unsqueeze(1)  # Shape: [BATCH_SIZE, 1]

        # Evaluate Q-value using the target Q-network (Double DQN)
        next_q_values = self.Q_cap(next_state_batch).gather(1, next_actions).squeeze(1)  # Shape: [BATCH_SIZE]
        next_q_values = next_q_values.detach()  # Detach to prevent gradients flowing into the target network

        # Compute the target Q-value
        expected_q_values = reward_batch + (GAMMA * next_q_values)

        # Compute the loss using SmoothL1Loss (Huber Loss)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Log the loss to wandb
        wandb.log({"loss": loss.item()})

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()

        # Update the target network periodically
        if self.training_step % TARGET_UPDATE_FREQUENCY == 0:
            self.Q_cap.load_state_dict(self.Q.state_dict())

        self.training_step += 1
        ###########################
