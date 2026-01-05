import torch
import torch.nn as nn
import numpy as np
import os
import glob
import sys

# config
DATA_DIR = "potential_categories"
MODEL_PATH = "anyon_brain.pth"
OUTPUT_LOG = "full_inference_report.txt"
RANK = 4

# model match
class AnyonNet(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rank**3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, rank)
        )
    def forward(self, x): return self.net(x)

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
        return torch.zeros(N_tensor.shape[0]).float()

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if not files:
        print("No files found.")
        return

    print(f"Loading Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    model = AnyonNet(RANK)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()

    print(f"Processing {len(files)} files. Writing results to {OUTPUT_LOG}...")
    
    # Log File
    with open(OUTPUT_LOG, "w") as log_file:
        log_file.write(f"filename, D_total_true, D_total_pred, error, raw_pred_vector\n")
        
        total_error = 0
        valid_count = 0

        with torch.no_grad():
            for i, fpath in enumerate(files):
                try:
                    data = torch.load(fpath, map_location='cpu', weights_only=False)
                    N = data['N'].float()
                    
                    if N.shape[0] != RANK: continue

                    if 'd' in data:
                        d_true = data['d'].float()
                    else:
                        d_true = calculate_d_from_N(N)

                    d_pred = model(N.flatten())

                    # Metrics
                    D_true = torch.sqrt(torch.sum(d_true**2)).item()
                    D_pred = torch.sqrt(torch.sum(d_pred**2)).item()
                    
                    error = abs(D_true - D_pred)
                    total_error += error
                    valid_count += 1

                    fname = os.path.basename(fpath)
                    vec_str = str(np.round(d_pred.numpy(), 2)).replace('\n', '')
                    
                    log_line = f"{fname}, {D_true:.4f}, {D_pred:.4f}, {error:.6f}, \"{vec_str}\"\n"
                    log_file.write(log_line)
                    # just incase 
                
                    if i % 100 == 0:
                        print(f"Processed {i}/{len(files)}...")

                except Exception as e:
                    print(f"Skipping {fpath}: {e}")

        # Summary
        avg_err = total_error / valid_count if valid_count > 0 else 0
        summary = f"\n--- COMPLETED ---\nProcessed: {valid_count}\nAverage Error: {avg_err:.6f}\n"
        print(summary)
        log_file.write(summary)

if __name__ == "__main__":
    main()