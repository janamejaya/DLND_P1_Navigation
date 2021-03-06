## Report

The objective of the project was to train an agent to successfully navigate the banana environment 

### Learning Algorithm

Since the number of actions is finite, the learning algorithm employed was [Deep Q-learning](https://www.nature.com/articles/nature14236) with the following modification
- A [dueling Q-network](https://arxiv.org/abs/1511.06581) architecture was used instead of the usual single-head neural network that outputs the action-value function
- The mean squared error loss was used instead of the Huber-loss used for loss-clipping.

### Neural network architecture
The following dueling Q-network architecture was employed
- The first two 2 hidden layers have 32 and 16 neurons, the output of which is connected to two branches, one each for the state and advantage functions outputs that are finally combined to generate the action-value for each action.
- Each branch has one hidden layer with 4 neurons connected to an output layer with the appropriate number of neurons for state and advantage functions.
- The maximum value of the advantage function was subtracted from the advantage function outputs before adding the resultant to the value function output

This could be approximately summarized as:

[State] -> [32] -> [16] -> [4] -> [1] ----------------------

                   |                                |->[Number of actions]
                     
                  [4] -> [Number_of_actions] --------
                     

### Parameters and Hyperparameter

#### Rewards
- Discount factor: 0.99

#### Replay buffer
- Buffer size : 100,000
- Buffer update frequency: every 4 steps
- Batch size drawn from buffer: 64

#### Neural Network hyperparameters for both networks
- Optimizer: Adam
- Learning rate for Optimizer: 0.0005
- Update target network every : 4 steps

#### Exprimental setup
- Maximum number of episodes per experiment: 2000
- Maximum number of steps per episode : 1000
- Starting epsilon - 1.00
- Final epsilon - 0.01
- Epsilon decay factor - 0.995
- Number of experiments - 10

### Results
A successful agent was defined to be one that achieves a reward of 13 or more averaged over the last 100 episodes of an experiment. The number of episodes and total time required to successfully complete the experiment was noted in order to get an quantify the success.

Since neural network optimization and expoloration of the environment are stochastic processes, the experiment was repeated 10 times. The following average metrics were calculated when the experiments successfully completed.

1. Number of episodes required for completion - 517.10 +/- 35.73
2. Time (in seconds) required for completion - 814.99 +/- 57.70

The success of the agent for each of these 10 experiments is displayed in the following ![figure](https://github.com/janamejaya/DLND_P1_Navigation/blob/main/result_score.jpg)

### Future improvements
The agents performance can be improved by including prioritized replay buffers, add noise to the network, including multi-step updates to the value function, and use a distribution of rewards as incorporated in [Rainbow](https://arxiv.org/abs/1710.02298)
