"""NACA GNN - Graph Neural Network for NACA Airfoil Predictions."""

from .airfoil import generate_naca_airfoil, airfoil_to_graph
from .model import NACAGNN
from .predict import predict_airfoil_properties

__version__ = "0.1.0"
__all__ = [
    "generate_naca_airfoil",
    "airfoil_to_graph",
    "NACAGNN",
    "predict_airfoil_properties",
]
