"""Training utilities for NACA GNN model."""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np


def train_model(model, train_data, val_data=None, epochs=100, batch_size=32, 
                learning_rate=0.001, device='cpu', verbose=True):
    """
    Train the NACA GNN model.
    
    Args:
        model: NACAGNN model instance
        train_data: List of training Data objects
        val_data: List of validation Data objects (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
        
    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    if val_data is not None:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            # Reshape target to match batch
            target = batch.y.view(-1, 3)
            loss = F.mse_loss(out, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
        
        train_loss /= len(train_data)
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_data is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    target = batch.y.view(-1, 3)
                    loss = F.mse_loss(out, target)
                    val_loss += loss.item() * batch.num_graphs
            
            val_loss /= len(val_data)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}')
        else:
            scheduler.step(train_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}')
    
    return model, history


def evaluate_model(model, test_data, batch_size=32, device='cpu'):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained NACAGNN model
        test_data: List of test Data objects
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            
            all_preds.append(out.cpu())
            all_targets.append(batch.y.view(-1, 3).cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(mse)
    
    # Per-property metrics
    property_names = ['Lift Coefficient', 'Drag Coefficient', 'Thickness Position']
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'predictions': all_preds,
        'targets': all_targets,
    }
    
    for i, name in enumerate(property_names):
        metrics[f'{name}_mae'] = np.mean(np.abs(all_preds[:, i] - all_targets[:, i]))
    
    return metrics
