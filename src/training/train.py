import torch
from torch.utils.data import DataLoader, random_split
import os
import sys
from tqdm import tqdm
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
import json
from models.SRCNN import SRCNN
from data.dataset import DIV2KDataset
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Hyperparams
batch_size = 16
lr = 1e-4
epochs = 50

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
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Metrics
history = {
    "train_loss": [], 
    "val_loss": [], 
    "val_psnr": [], 
    "val_ssim": []
    }

os.makedirs("logs", exist_ok=True)

# Training loop simple
for epoch in range(epochs):
    model.train()
    
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                      unit="batch", leave=False)
    
    train_loss = 0.0
    for batch_idx, (lr_p, hr_p) in enumerate(train_loop):

        lr_p = lr_p.to(device)
        hr_p = hr_p.to(device)

        optimizer.zero_grad()
        sr = model(lr_p)
        loss = criterion(sr, hr_p)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        train_loop.set_postfix(loss=loss.item())
    
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                       unit="batch", leave=False)
        
        for lr_p, hr_p in val_loop:

            lr_p = lr_p.to(device)
            hr_p = hr_p.to(device)

            sr = model(lr_p)
            loss = criterion(sr, hr_p)
            val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())

            # Convert to numpy
            sr_np = sr.clamp(0,1).cpu().numpy()
            hr_np = hr_p.cpu().numpy()
            for i in range(sr_np.shape[0]):  # iterate over batch
                sr_img = np.transpose(sr_np[i], (1,2,0))
                hr_img = np.transpose(hr_np[i], (1,2,0))
                val_psnr += psnr(hr_img, sr_img, data_range=1.0)
                val_ssim += ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)            
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_ds)
    avg_val_ssim = val_ssim / len(val_ds)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

    history["train_loss"].append(float(avg_train_loss))
    history["val_loss"].append(float(avg_val_loss))
    history["val_psnr"].append(float(avg_val_psnr))
    history["val_ssim"].append(float(avg_val_ssim))

    with open("src/logs/srcnn_history.json", "w") as f:
        json.dump(history, f, indent=4)    

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/srcnn_baseline.pth")
