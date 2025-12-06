import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) model for regression.
    This model maps input features to a single continuous output value.
    """
    
    def __init__(self, input_dim):
        """
        Initialize the MLP architecture.
        
        Args:
            input_dim (int): Number of input features.
        """
        super().__init__()
        
        # Define a 3-layer fully connected neural network
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),   # Input → Hidden layer 1
            nn.ReLU(),                 # Activation function
            nn.Linear(64, 32),         # Hidden layer 1 → Hidden layer 2
            nn.ReLU(),                 # Activation function
            nn.Linear(32, 1)           # Hidden layer 2 → Output (regression)
        )

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor: Predicted continuous value.
        """
        return self.model(x)
