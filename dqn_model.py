#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # This line stores the number of possible actions in the class instance variable num_actions.
        self.num_actions = num_actions
         
        # 3 convolutional layers with ReLU activation functions. 
        # The convolutional layers are designed to process the input frames (states) and extract features. 
        self.network=nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Height and Width of the feature maps after the convolutional layers 
        h = w = self.conv_size(9,3,1)

        # Calculate the total size of the flattened input to the fully connected layers.
        self.input_sz = int(h*w*64)

        # 
        self.qvals = nn.Sequential(
            nn.Linear(self.input_sz, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions))
        
    # This static method computes the size of the feature map 
    # after a convolutional layer based on the input size, kernel size, and stride.

    @staticmethod
    def conv_size(size,kernel_size,stride):
        s=(size-(kernel_size-1)-1) / stride + 1
        return s 


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x=x.to(device)
        x=self.network(x)
        x=x.view(x.size(0),-1)
        Qvals=self.qvals(x)
        return Qvals
        ###########################
        # return x
