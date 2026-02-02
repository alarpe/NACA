"""NACA airfoil geometry generation and graph conversion utilities."""

import numpy as np
import torch
from torch_geometric.data import Data


def generate_naca_airfoil(naca_code, num_points=100):
    """
    Generate NACA 4-digit airfoil coordinates.
    
    Args:
        naca_code: String or int, e.g., '2412' or 2412
        num_points: Number of points to generate on the airfoil
        
    Returns:
        x, y_upper, y_lower: Arrays of x coordinates and upper/lower y coordinates
    """
    naca_str = str(naca_code).zfill(4)
    
    # Extract parameters
    m = int(naca_str[0]) / 100.0  # Maximum camber
    p = int(naca_str[1]) / 10.0   # Location of maximum camber
    t = int(naca_str[2:4]) / 100.0  # Maximum thickness
    
    # Generate x coordinates (cosine spacing for better resolution at leading edge)
    beta = np.linspace(0, np.pi, num_points)
    x = (1 - np.cos(beta)) / 2
    
    # Thickness distribution
    yt = 5 * t * (
        0.2969 * np.sqrt(x) - 
        0.1260 * x - 
        0.3516 * x**2 + 
        0.2843 * x**3 - 
        0.1015 * x**4
    )
    
    # Camber line
    if p == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2)
        )
        dyc_dx = np.where(
            x < p,
            2 * m / p**2 * (p - x),
            2 * m / (1 - p)**2 * (p - x)
        )
    
    # Angle
    theta = np.arctan(dyc_dx)
    
    # Upper and lower surfaces
    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)
    
    return x, y_upper, y_lower, x_upper, y_upper, x_lower, y_lower


def airfoil_to_graph(x_upper, y_upper, x_lower, y_lower):
    """
    Convert airfoil coordinates to a graph structure for GNN.
    
    Args:
        x_upper, y_upper: Upper surface coordinates
        x_lower, y_lower: Lower surface coordinates
        
    Returns:
        torch_geometric.data.Data: Graph data object
    """
    # Combine upper and lower surface points
    x_coords = np.concatenate([x_upper, x_lower[::-1]])
    y_coords = np.concatenate([y_upper, y_lower[::-1]])
    
    # Node features: [x, y] coordinates
    num_nodes = len(x_coords)
    node_features = np.column_stack([x_coords, y_coords])
    
    # Create edges: connect consecutive points and add some cross-connections
    edges = []
    
    # Sequential edges (forming the airfoil contour)
    for i in range(num_nodes):
        edges.append([i, (i + 1) % num_nodes])
        edges.append([(i + 1) % num_nodes, i])
    
    # Add cross-connections for better information flow
    # Connect points to their k-nearest neighbors
    k = 5
    for i in range(num_nodes):
        distances = np.sqrt(
            (x_coords - x_coords[i])**2 + 
            (y_coords - y_coords[i])**2
        )
        nearest = np.argsort(distances)[1:k+1]  # Skip self (distance=0)
        for j in nearest:
            edges.append([i, j])
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


def generate_sample_data(num_samples=100, num_points=50):
    """
    Generate sample dataset of NACA airfoils with random parameters.
    
    Args:
        num_samples: Number of airfoils to generate
        num_points: Number of points per airfoil
        
    Returns:
        List of (graph, properties) tuples
    """
    dataset = []
    
    for _ in range(num_samples):
        # Random NACA 4-digit parameters
        m = np.random.randint(0, 10)  # Max camber (0-9%)
        p = np.random.randint(0, 10)  # Position (0-90% of chord)
        t = np.random.randint(6, 21)  # Thickness (6-20%)
        
        naca_code = f"{m}{p}{t:02d}"
        
        # Generate airfoil
        x, y_u, y_l, x_u, y_u, x_l, y_l = generate_naca_airfoil(
            naca_code, num_points
        )
        
        # Convert to graph
        graph = airfoil_to_graph(x_u, y_u, x_l, y_l)
        
        # Simplified aerodynamic properties (dummy values for demonstration)
        # In practice, these would come from CFD simulations or experiments
        # Properties: [lift_coefficient, drag_coefficient, max_thickness_pos]
        cl = 0.1 * m + 0.05  # Simplified lift coefficient
        cd = 0.005 + 0.0001 * t  # Simplified drag coefficient
        thickness_pos = p / 10.0  # Max thickness position
        
        properties = torch.tensor([cl, cd, thickness_pos], dtype=torch.float)
        graph.y = properties
        
        dataset.append(graph)
    
    return dataset
