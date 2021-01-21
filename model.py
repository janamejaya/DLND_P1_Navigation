import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Action-value function approximation multi-layer perceptron."""

    def __init__(self, state_size, action_size, seed, hidden_layers, head_name, head_scale):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int):
                number of nodes for each hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.head_name=head_name
        self.head_scale=head_scale
        
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        
        if self.head_name=='DQN':
            self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
            self.fc4 = nn.Linear(hidden_layers[2], action_size)
        elif self.head_name=='DuelingDQN':
            half_split = int(hidden_layers[2]/2)
            
            # Value function split
            self.fc3_1 = nn.Linear(hidden_layers[1], half_split)
            self.fc4_1 = nn.Linear(half_split, 1)
            
            # Advantage function split
            self.fc3_2 = nn.Linear(hidden_layers[1], half_split)
            self.fc4_2 = nn.Linear(half_split, action_size)
            
    def forward(self, state):
        out1 = F.relu(self.fc1(state))
        out2 = F.relu(self.fc2(out1))
        
        if self.head_name=='DQN':
            out3 = F.relu(self.fc3(out2))
            return self.fc4(out3)
        elif self.head_name=='DuelingDQN':
            # The value function
            out3_1 = self.fc3_1(out2)
            self.V = self.fc4_1(out3_1)
            #print('self.V.shape = ',self.V.shape)
            
            # The advantage function
            out3_2 = self.fc3_2(out2)
            self.A = self.fc4_2(out3_2)
            #print('self.A.shape = ',self.A.shape)
            
            # Reshape the value function to resemble the shape of the action function
            self.V = self.V.expand_as(self.A)
            #print('after expansion: self.V.shape = ',self.V.shape)

            # If scale for advantage function is set by the maximum value of advantage
            if self.head_scale=='max':             
                self.maxA = torch.max(self.A, 1, keepdim=True)[0].expand_as(self.A)
                return self.V + self.A - self.maxA
            # If scale for advantage function is set by the mean value of advantage
            elif self.head_scale=='mean':
                self.meanA = torch.mean(self.A, 1, keepdim=True).expand_as(self.A)
                return self.V + self.A - self.meanA
            # If no scale adjustment is required
            elif self.head_scale==None:
                return self.V + self.A

class QNetwork_no_layer_per_split_head(nn.Module):
    """Action-value approximation network."""

    def __init__(self, state_size, action_size, seed, hidden_layers, head_name, head_scale):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int):
                number of nodes for each hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.head_name=head_name
        self.head_scale=head_scale
        
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        
        if self.head_name=='DQN':
            self.fc3 = nn.Linear(hidden_layers[1], action_size)
        elif self.head_name=='DuelingDQN':
            self.fc3_1 = nn.Linear(hidden_layers[1], 1) #V(s)
            self.fc3_2 = nn.Linear(hidden_layers[1], action_size) # A(s)


    def forward(self, state):
        out1 = F.relu(self.fc1(state))
        out2 = F.relu(self.fc2(out1))
        
        if self.head_name=='DQN':
            #print(self.fc3(out2).shape)
            return self.fc3(out2)
        elif self.head_name=='DuelingDQN':
            # The value function
            self.V = self.fc3_1(out2)
            self.A = self.fc3_2(out2)
            self.V = self.V.expand_as(self.A)

            # If scale for advantage function is set by the maximum value of advantage
            if self.head_scale=='max':             
                self.maxA = torch.max(self.A, 1, keepdim=True)[0].expand_as(self.A)
                return self.V + self.A - self.maxA
            # If scale for advantage function is set by the mean value of advantage
            elif self.head_scale=='mean':
                self.meanA = torch.mean(self.A, 1, keepdim=True).expand_as(self.A)
                return self.V + self.A - self.meanA
            # If no scale adjustment is required
            elif self.head_scale==None:
                return self.V + self.A


