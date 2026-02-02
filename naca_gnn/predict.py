"""Prediction utilities for NACA airfoils using trained GNN model."""

import torch
from .airfoil import generate_naca_airfoil, airfoil_to_graph


def predict_airfoil_properties(model, naca_code, num_points=100, device='cpu'):
    """
    Predict aerodynamic properties for a NACA airfoil.
    
    Args:
        model: Trained NACAGNN model
        naca_code: NACA 4-digit code (e.g., '2412' or 2412)
        num_points: Number of points to discretize the airfoil
        device: Device to run prediction on ('cpu' or 'cuda')
        
    Returns:
        properties: Dictionary with predicted properties
            - 'lift_coefficient': Predicted lift coefficient
            - 'drag_coefficient': Predicted drag coefficient
            - 'thickness_position': Predicted max thickness position
    """
    model = model.to(device)
    model.eval()
    
    # Generate airfoil geometry
    _, _, _, x_upper, y_upper, x_lower, y_lower = generate_naca_airfoil(naca_code, num_points)
    
    # Convert to graph
    graph = airfoil_to_graph(x_upper, y_upper, x_lower, y_lower)
    graph = graph.to(device)
    
    # Make prediction
    with torch.no_grad():
        pred = model(graph)
    
    pred = pred.cpu().numpy()[0]
    
    properties = {
        'naca_code': str(naca_code),
        'lift_coefficient': float(pred[0]),
        'drag_coefficient': float(pred[1]),
        'thickness_position': float(pred[2]),
    }
    
    return properties


def batch_predict(model, naca_codes, num_points=100, device='cpu'):
    """
    Predict properties for multiple NACA airfoils.
    
    Args:
        model: Trained NACAGNN model
        naca_codes: List of NACA 4-digit codes
        num_points: Number of points to discretize each airfoil
        device: Device to run prediction on
        
    Returns:
        results: List of property dictionaries
    """
    results = []
    
    for naca_code in naca_codes:
        props = predict_airfoil_properties(model, naca_code, num_points, device)
        results.append(props)
    
    return results
