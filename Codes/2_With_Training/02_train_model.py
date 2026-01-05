import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import os
import numpy as np

# config
DATA_DIR = "potential_categories"  
BATCH_SIZE = 16  
LR = 0.001
EPOCHS = 200

def calculate_d_from_N(N_tensor):
    """
    Recalculates Quantum Dimensions if missing.
    """
    try:
        M = torch.sum(N_tensor, dim=0).cpu().numpy()
        vals, vecs = np.linalg.eig(M)
        max_idx = np.argmax(np.abs(vals))
        d = np.abs(vecs[:, max_idx])
        d = d / d[0] # Normalize
        return torch.tensor(d).float()
    except:
        return torch.ones(N_tensor.shape[0]).float()

# dataset
class FusionRingDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.data = []
        self.rank = None # will determine this from the first file found
        
        print(f"Looking for data in: {os.path.abspath(data_dir)}")
        print(f"Found {len(self.files)} files. Analyzing content...")
        
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in '{data_dir}'. Check folder name or run Solver first.")

        for f in self.files:
            try:
                payload = torch.load(f, map_location='cpu', weights_only=False)
                N = payload['N']
                self.rank = N.shape[0]
                print(f"--> Auto-detected Data Rank: {self.rank}")
                break
            except:
                continue
        
        if self.rank is None:
            raise RuntimeError("Could not read any files to determine Rank.")

        skipped_wrong_rank = 0
        skipped_corrupt = 0

        for f in self.files:
            try:
                payload = torch.load(f, map_location='cpu', weights_only=False)
                N = payload['N'].float()
                
                # Filter strictly by the detected rank
                if N.shape[0] != self.rank:
                    skipped_wrong_rank += 1
                    continue

                if 'd' in payload:
                    d = payload['d'].float()
                else:
                    d = calculate_d_from_N(N)

                self.data.append((N.flatten(), d))
                
            except Exception:
                skipped_corrupt += 1

        print(f"--- DATASET REPORT ---")
        print(f"Loaded:           {len(self.data)} valid samples (Rank {self.rank})")
        print(f"Skipped (Diff R): {skipped_wrong_rank}")
        print(f"Skipped (Error):  {skipped_corrupt}")
        
        if len(self.data) == 0:
            raise RuntimeError("Dataset is empty after filtering!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# model
class AnyonNet(nn.Module):
    def __init__(self, rank):
        super().__init__()
        input_size = rank ** 3  
        output_size = rank      
        
        print(f"Initializing Neural Net: Input={input_size} -> Output={output_size}")
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

# training
def main():
    try:
        full_dataset = FusionRingDataset(DATA_DIR)
        current_rank = full_dataset.rank 
    except RuntimeError as e:
        print(f"\nCRITICAL ERROR: {e}")
        return

    train_size = int(0.8 * len(full_dataset))
    if train_size == 0 and len(full_dataset) > 0: train_size = len(full_dataset)
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if test_size > 0:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        test_loader = None

    model = AnyonNet(current_rank)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() 

    print("\nRunning")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        count = 0
        for N_batch, d_batch in train_loader:
            optimizer.zero_grad()
            prediction = model(N_batch)
            loss = criterion(prediction, d_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
            
        if (epoch+1) % 20 == 0:
            avg_loss = total_loss / count if count > 0 else 0
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    if test_loader:
        print("\nTry on unseen data")
        model.eval()
        with torch.no_grad():
            for i, (N, true_d) in enumerate(test_loader):
                if i >= 3: break 
                
                pred_d = model(N)
                
                print(f"Ring #{i+1}:")
                print(f"  True D: {np.round(true_d.numpy()[0], 4)}")
                print(f"  AI Est: {np.round(pred_d.numpy()[0], 4)}")
                print("-" * 30)

    # Save
    torch.save(model.state_dict(), "anyon_brain.pth") # feel free to rename
    
    print("Model saved as 'anyon_brain.pth'")

if __name__ == "__main__":
    main()