#!/usr/bin/env python3
"""
Example training script for testing speedrun
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    print("Starting training...")
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for multiple GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create model
    model = SimpleModel().to(device)
    
    # Use DataParallel for multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Generate dummy data
    X = torch.randn(1000, 10).to(device)
    y = torch.randn(1000, 1).to(device)
    
    # Train
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Save model
    print("Saving model...")
    Path("models").mkdir(exist_ok=True)
    
    # Handle DataParallel model saving
    model_to_save = model.module if hasattr(model, 'module') else model
    
    torch.save(model_to_save.state_dict(), "models/simple_model.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'num_gpus_used': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }, "models/checkpoint.pth")
    
    print("Training completed!")

if __name__ == "__main__":
    train()