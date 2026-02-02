# NACA GNN - Graph Neural Network for Airfoil Prediction

A Graph Neural Network (GNN) based system for predicting aerodynamic properties of NACA airfoils. This project uses PyTorch Geometric to model airfoil geometries as graphs and predict key aerodynamic characteristics.

## Features

- **NACA 4-digit airfoil generation**: Generate airfoil geometries from NACA codes
- **Graph-based representation**: Convert airfoil geometries to graph structures
- **GNN prediction model**: Deep learning model for property prediction
- **Training utilities**: Complete training pipeline with validation
- **Easy-to-use prediction interface**: Predict properties for any NACA airfoil
- **Visualization tools**: Plot and compare airfoil shapes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alarpe/NACA.git
cd NACA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model

Train a GNN model on randomly generated NACA airfoils:

```bash
python examples/train_and_predict.py
```

This will:
- Generate 500 sample NACA airfoils
- Train a GNN model to predict aerodynamic properties
- Evaluate the model on test data
- Save the trained model to `naca_gnn_model.pth`

### 2. Make Predictions

Predict properties for a specific NACA airfoil:

```bash
python examples/predict.py 2412
```

Example output:
```
Predicted Properties:
  Lift Coefficient:     0.2500
  Drag Coefficient:     0.0065
  Thickness Position:   0.4000
```

### 3. Visualize Airfoils

Visualize NACA airfoil shapes:

```bash
python examples/visualize.py
```

## Usage

### Python API

```python
import torch
from naca_gnn.airfoil import generate_naca_airfoil, generate_sample_data
from naca_gnn.model import NACAGNN
from naca_gnn.train import train_model
from naca_gnn.predict import predict_airfoil_properties

# Generate sample data
dataset = generate_sample_data(num_samples=100, num_points=50)

# Create and train model
model = NACAGNN(input_dim=2, hidden_dim=64, output_dim=3)
model, history = train_model(model, dataset, epochs=50)

# Make predictions
properties = predict_airfoil_properties(model, "2412")
print(f"Lift coefficient: {properties['lift_coefficient']:.4f}")
```

### Generate NACA Airfoil

```python
from naca_gnn.airfoil import generate_naca_airfoil

# Generate NACA 2412 airfoil
x, y_u, y_l, x_u, y_u, x_l, y_l = generate_naca_airfoil("2412", num_points=100)
```

### Convert to Graph

```python
from naca_gnn.airfoil import airfoil_to_graph

# Convert airfoil coordinates to graph
graph = airfoil_to_graph(x_u, y_u, x_l, y_l)
```

## Model Architecture

The NACAGNN model uses:
- **Input**: 2D node features (x, y coordinates)
- **Graph Convolution**: 3 layers of GCN with batch normalization
- **Pooling**: Global mean pooling for graph-level representation
- **Output**: 3 predicted properties
  - Lift coefficient
  - Drag coefficient
  - Maximum thickness position

## NACA 4-Digit Airfoils

The NACA 4-digit series describes the airfoil geometry using 4 digits:
- **First digit**: Maximum camber (% of chord)
- **Second digit**: Position of maximum camber (in 1/10 of chord)
- **Last two digits**: Maximum thickness (% of chord)

Example: NACA 2412
- 2% maximum camber
- Located at 40% chord
- 12% maximum thickness

## Project Structure

```
NACA/
├── naca_gnn/
│   ├── __init__.py          # Package initialization
│   ├── airfoil.py           # Airfoil generation and graph conversion
│   ├── model.py             # GNN model architecture
│   ├── train.py             # Training utilities
│   └── predict.py           # Prediction functions
├── examples/
│   ├── train_and_predict.py # Complete training example
│   ├── predict.py           # Simple prediction script
│   └── visualize.py         # Visualization utilities
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- SciPy >= 1.10.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- NACA airfoil equations from NASA technical reports
- PyTorch Geometric for graph neural network framework