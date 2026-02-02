"""Simple prediction script using a pre-trained model."""

import torch
import sys
from naca_gnn.model import NACAGNN
from naca_gnn.predict import predict_airfoil_properties


def main():
    """Load model and make predictions."""
    # Check if NACA code provided
    if len(sys.argv) < 2:
        print("Usage: python predict.py <naca_code> [model_path]")
        print("Example: python predict.py 2412")
        print("\nUsing default NACA 2412 for demonstration...")
        naca_code = "2412"
    else:
        naca_code = sys.argv[1]
    
    # Model path
    model_path = sys.argv[2] if len(sys.argv) > 2 else "naca_gnn_model.pth"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load model
    print(f"Loading model from {model_path}...")
    model = NACAGNN(input_dim=2, hidden_dim=64, output_dim=3, num_layers=3)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please train a model first using examples/train_and_predict.py")
        return
    
    # Make prediction
    print(f"\nPredicting properties for NACA {naca_code}...")
    properties = predict_airfoil_properties(model, naca_code, device=device)
    
    print("\nPredicted Properties:")
    print(f"  Lift Coefficient:     {properties['lift_coefficient']:.4f}")
    print(f"  Drag Coefficient:     {properties['drag_coefficient']:.4f}")
    print(f"  Thickness Position:   {properties['thickness_position']:.4f}")


if __name__ == "__main__":
    main()
