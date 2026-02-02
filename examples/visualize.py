"""Visualize NACA airfoils and their properties."""

import matplotlib.pyplot as plt
import numpy as np
from naca_gnn.airfoil import generate_naca_airfoil


def plot_airfoil(naca_code, save_path=None):
    """
    Plot a NACA airfoil.
    
    Args:
        naca_code: NACA 4-digit code
        save_path: Path to save the plot (optional)
    """
    # Generate airfoil
    x, y_u, y_l, x_u, y_u, x_l, y_l = generate_naca_airfoil(naca_code, num_points=100)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot upper and lower surfaces
    ax.plot(x_u, y_u, 'b-', linewidth=2, label='Upper surface')
    ax.plot(x_l, y_l, 'r-', linewidth=2, label='Lower surface')
    
    # Fill between surfaces
    ax.fill_between(x_u, y_u, y_l[::-1], alpha=0.3, color='gray')
    
    # Formatting
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel('y/c', fontsize=12)
    ax.set_title(f'NACA {naca_code} Airfoil', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_airfoils(naca_codes, save_path=None):
    """
    Plot multiple NACA airfoils for comparison.
    
    Args:
        naca_codes: List of NACA 4-digit codes
        save_path: Path to save the plot (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(naca_codes)))
    
    for i, naca_code in enumerate(naca_codes):
        x, y_u, y_l, x_u, y_u, x_l, y_l = generate_naca_airfoil(naca_code, num_points=100)
        
        # Plot airfoil
        ax.plot(x_u, y_u, '-', color=colors[i], linewidth=2, 
                label=f'NACA {naca_code}')
        ax.plot(x_l, y_l, '-', color=colors[i], linewidth=2)
    
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel('y/c', fontsize=12)
    ax.set_title('NACA Airfoil Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Visualizing NACA airfoils...")
    
    # Plot single airfoil
    print("\n1. Plotting NACA 2412...")
    plot_airfoil("2412", "naca_2412.png")
    
    # Plot multiple airfoils
    print("\n2. Plotting comparison of multiple airfoils...")
    airfoils = ["0012", "2412", "4415", "6409"]
    plot_multiple_airfoils(airfoils, "naca_comparison.png")
    
    print("\nVisualization complete!")
