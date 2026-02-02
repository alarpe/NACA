"""Graph Neural Network model for NACA airfoil property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class NACAGNN(nn.Module):
    """
    Graph Neural Network for predicting aerodynamic properties of NACA airfoils.
    
    Architecture:
    - Multiple graph convolutional layers
    - Global pooling to aggregate node features
    - Fully connected layers for prediction
    """
    
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=3, num_layers=3):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of node features (default: 2 for x,y coordinates)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (number of properties to predict)
            num_layers: Number of graph convolutional layers
        """
        super(NACAGNN, self).__init__()
        
        self.num_layers = num_layers
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: torch_geometric.data.Data or Batch object
            
        Returns:
            predictions: Tensor of predicted properties
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )
        
        # Graph convolutional layers with batch normalization and ReLU
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, data):
        """
        Make predictions (inference mode).
        
        Args:
            data: torch_geometric.data.Data or Batch object
            
        Returns:
            predictions: Tensor of predicted properties
        """
        self.eval()
        with torch.no_grad():
            return self.forward(data)
