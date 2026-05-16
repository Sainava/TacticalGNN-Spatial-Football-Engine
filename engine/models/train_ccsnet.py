import torch
import torch.nn as nn
import time
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import os
import sys

# Import your custom architecture
from gnn_architecture import CCSNet

def train_model():
    # 1. HARDWARE TARGETING
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon (MPS) detected. Engaging M3 GPU...")
    else:
        device = torch.device("cpu")
        print("MPS not available. Falling back to CPU...")

    # 2. LOAD AND SPLIT DATASET
    print("Loading tensor dataset into memory...")
    data_path = "../../data/processed/ccsn_tensors.pt"
    if not os.path.exists(data_path):
        print("Dataset not found! Please run ccsn_dataset.py first.")
        return
        
    full_dataset = torch.load(data_path, weights_only=False)
    
    # 70% Training, 30% Validation
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset Split: {len(train_dataset)} Train | {len(val_dataset)} Validation")

    # 3. INITIALIZE MODEL AND OPTIMIZER
    model = CCSNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    num_epochs = 50
    model_save_path = "../../models/ccsnet_best_model.pth"
    os.makedirs("../../models", exist_ok=True)

    # 4. TRAINING LOOP
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch in train_pbar:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_idx = batch.batch.to(device)
            y = batch.y.to(device).view(-1, 1)

            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr, batch_idx)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{(train_correct/train_total)*100:.1f}%'})

        # 5. VALIDATION PHASE
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                edge_attr = batch.edge_attr.to(device)
                batch_idx = batch.batch.to(device)
                y = batch.y.to(device).view(-1, 1)
                
                out = model(x, edge_index, edge_attr, batch_idx)
                preds = (torch.sigmoid(out) > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = (val_correct / val_total) * 100
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # 6. SAVE BEST MODEL
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with accuracy: {best_val_acc:.2f}%")

    print("\nTraining Complete.")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()