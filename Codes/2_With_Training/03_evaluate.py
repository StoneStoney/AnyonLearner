import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# config
DATA_DIR = "potential_categories"
MODEL_PATH = "anyon_brain.pth"
OUTPUT_IMAGE = "prediction_results.png"

# Model
class AnyonNet(nn.Module):
    def __init__(self, rank):
        super().__init__()
        input_size = rank ** 3  
        output_size = rank      
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)
  
# calculate d from N
def calculate_d_from_N(N_tensor):
    try:
        M = torch.sum(N_tensor, dim=0).cpu().numpy()
        vals, vecs = np.linalg.eig(M)
        max_idx = np.argmax(np.abs(vals))
        d = np.abs(vecs[:, max_idx])
        d = d / d[0]
        return torch.tensor(d).float()
    except:
        return torch.ones(N_tensor.shape[0]).float()

# main
def main():                                                   
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if not files:
        print(f"No data found in {DATA_DIR}")
        return

    first_payload = torch.load(files[0], map_location='cpu', weights_only=False)
    rank = first_payload['N'].shape[0]
    print(f"Detected Rank: {rank}")

    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found. Run training first!")
        return
        
    model = AnyonNet(rank)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()

    true_d_totals = []
    pred_d_totals = []
    
    print(f"Processing {len(files)} samples...")

    with torch.no_grad():
        for f in files:
            try:
                data = torch.load(f, map_location='cpu', weights_only=False)
                N = data['N'].float()
                
                if N.shape[0] != rank: continue

                if 'd' in data:
                    d_true = data['d'].float()
                else:
                    d_true = calculate_d_from_N(N)

                d_pred = model(N.flatten())

                D_total_true = torch.sqrt(torch.sum(d_true**2)).item()
                D_total_pred = torch.sqrt(torch.sum(d_pred**2)).item()

                true_d_totals.append(D_total_true)
                pred_d_totals.append(D_total_pred)
            except:
                pass

    plt.figure(figsize=(10, 6))
    plt.scatter(true_d_totals, pred_d_totals, alpha=0.6, c='blue', edgecolors='k')
    
    min_val = min(min(true_d_totals), min(pred_d_totals))
    max_val = max(max(true_d_totals), max(pred_d_totals))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")

    plt.xlabel("Actual Total Dimension (D)")
    plt.ylabel("Predicted Dimension (D)")
    plt.title(f"Performance on Topological Quantum Field Theories (Rank {rank})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(OUTPUT_IMAGE)
    print(f"Plot saved to {OUTPUT_IMAGE}")
    
    plt.show()

if __name__ == "__main__":
    main()