import torch
from torch.utils.data import DataLoader, random_split
import os
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from models.SRCNN import SRCNN
from data.dataset import DIV2KDataset
import torch.nn as nn
import torch.optim as optim

# Hyperparams
batch_size = 16
lr = 1e-4
epochs = 5

# Dataset
dataset = DIV2KDataset(hr_dir="data/raw/DIV2K/DIV2K_train_HR",
                        lr_dir="data/processed/DIV2K/DIV2K_train_LR_x2",
                        patch_size=64)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Model
model = SRCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop simple
for epoch in range(epochs):
    model.train()
    
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                      unit="batch", leave=False)
    
    train_loss = 0.0
    for batch_idx, (lr_p, hr_p) in enumerate(train_loop):
        optimizer.zero_grad()
        sr = model(lr_p)
        loss = criterion(sr, hr_p)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        train_loop.set_postfix(loss=loss.item())
    
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                       unit="batch", leave=False)
        
        for lr_p, hr_p in val_loop:
            sr = model(lr_p)
            loss = criterion(sr, hr_p)
            val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/srcnn_baseline.pth")
