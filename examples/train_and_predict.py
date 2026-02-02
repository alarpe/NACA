"""Example script demonstrating training and prediction with NACA GNN."""

import torch
import numpy as np
from naca_gnn.airfoil import generate_sample_data
from naca_gnn.model import NACAGNN
from naca_gnn.train import train_model, evaluate_model
from naca_gnn.predict import predict_airfoil_properties, batch_predict


def main():
    """Run complete training and prediction example."""
    print("=" * 60)
    print("NACA GNN - Airfoil Property Prediction Example")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Generate sample dataset
    print("\n1. Generating sample dataset...")
    print("   Creating 500 random NACA airfoils...")
    dataset = generate_sample_data(num_samples=500, num_points=50)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create model
    print("\n2. Creating GNN model...")
    model = NACAGNN(
        input_dim=2,
        hidden_dim=64,
        output_dim=3,
        num_layers=3
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n3. Training model...")
    model, history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        device=device,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    metrics = evaluate_model(model, test_data, device=device)
    print(f"   Test MSE: {metrics['mse']:.6f}")
    print(f"   Test MAE: {metrics['mae']:.6f}")
    print(f"   Test RMSE: {metrics['rmse']:.6f}")
    print("\n   Per-property MAE:")
    print(f"   - Lift Coefficient: {metrics['Lift Coefficient_mae']:.6f}")
    print(f"   - Drag Coefficient: {metrics['Drag Coefficient_mae']:.6f}")
    print(f"   - Thickness Position: {metrics['Thickness Position_mae']:.6f}")
    
    # Save model
    print("\n5. Saving model...")
    torch.save(model.state_dict(), 'naca_gnn_model.pth')
    print("   Model saved to 'naca_gnn_model.pth'")
    
    # Example predictions
    print("\n6. Example predictions on specific NACA airfoils...")
    example_airfoils = ['0012', '2412', '4415', '6409']
    
    predictions = batch_predict(model, example_airfoils, device=device)
    
    for pred in predictions:
        print(f"\n   NACA {pred['naca_code']}:")
        print(f"      Lift Coefficient:  {pred['lift_coefficient']:.4f}")
        print(f"      Drag Coefficient:  {pred['drag_coefficient']:.4f}")
        print(f"      Thickness Position: {pred['thickness_position']:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
