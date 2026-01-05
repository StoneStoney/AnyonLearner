import os
os.environ["OMP_NUM_THREADS"] = "8"  # This can be customized, I'm on a personal device so 8 is fine. 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import requests
import zipfile
import io
import shutil
from tqdm import tqdm 

# Config
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "levin_wen_solutions"

# using the github linked
def download_data(root_dir):
    if os.path.exists(root_dir) and len(os.listdir(root_dir)) > 0: return
    url = "https://github.com/JCBridgeman/smallRankUnitaryFusionData/archive/refs/heads/main.zip"
    try:
        r = requests.get(url); z = zipfile.ZipFile(io.BytesIO(r.content)); z.extractall(".")
        extracted = next(n for n in os.listdir(".") if n.startswith("smallRankUnitaryFusionData-"))
        if os.path.exists(root_dir): shutil.rmtree(root_dir)
        os.rename(extracted, root_dir)
    except: pass

class AnyonDataset(Dataset): # Using AnyonWiki data
    def __init__(self, root_dir):
        self.rings = []
        download_data(root_dir)
        for dirpath, _, filenames in os.walk(root_dir):
            if "Nabc.txt" in filenames:
                try: self.rings.append(self.parse(os.path.join(dirpath, "Nabc.txt")))
                except: pass

    def parse(self, fpath):
        data = np.loadtxt(fpath, dtype=int)
        if data.size > 0 and np.min(data[:, :3]) == 1: data[:, :3] -= 1
        rank = int(np.max(data[:, :3])) + 1
        N = torch.zeros((rank, rank, rank))
        for r in data: N[int(r[0]), int(r[1]), int(r[2])] = r[3]
        return {"name": os.path.basename(os.path.dirname(fpath)), "N": N, "dim": rank}
    
    def __len__(self): return len(self.rings)
    def __getitem__(self, idx): return self.rings[idx]

# simpel solver
class LevinWenSolver(nn.Module):
    def __init__(self, N_tensor):
        super().__init__()
        self.register_buffer('N', N_tensor)
        self.dim = N_tensor.shape[0]

        # Solve Dimensions (d_i)
        M = torch.sum(self.N, dim=0).cpu().numpy()
        vals, vecs = np.linalg.eig(M)
        d = np.abs(vecs[:, np.argmax(np.abs(vals))])
        self.register_buffer('d', torch.tensor(d / d[0], device=DEVICE))
        self.D_total = torch.sqrt(torch.sum(self.d**2))
        self.register_buffer('mask', self._build_mask())
        
        # Learnable F-Symbols
        self.F_params = nn.Parameter(torch.randn(*self.mask.shape, 2) * 0.1)

    def _build_mask(self):
        mask = torch.zeros((self.dim,)*6)
        N = self.N.cpu().numpy()
        import itertools
        for i,j,k,l,m,n in itertools.product(range(self.dim), repeat=6):
            if (N[i,j,m] and N[m,k,l] and N[j,k,n] and N[i,n,l]):
                mask[i,j,k,l, m,n] = 1.0
        return mask.to(DEVICE)

    def get_F(self):
        return torch.view_as_complex(self.F_params) * self.mask

    def loss(self):
        F = self.get_F()
        ortho = torch.einsum('ijklmn, ijklpn -> ijklmp', F, F.conj())
        valid_rows = (self.mask.sum(dim=5) > 0).float()
        return torch.norm(ortho - torch.diag_embed(valid_rows))

    def save(self, name):
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, f"{name}.pt")
        S_col0 = self.d / self.D_total
        torch.save({
            'name': name, 'rank': self.dim, 'N': self.N.cpu(),
            'F': self.get_F().detach().cpu(), 'd': self.d.cpu(),
            'S_col0': S_col0.cpu(), 'D': self.D_total.cpu()
        }, path)

# main pipeline
def main():
    dataset = AnyonDataset("smallRankUnitaryFusionData")
    print(f"\n--- Solving {len(dataset)} fusion rings ---")
    print(f"Output Directory: {os.path.abspath(SAVE_DIR)}\n")

    pbar = tqdm(range(len(dataset)), unit="ring")
    
    solved_count = 0
    
    for i in pbar:
        data = dataset[i]
        name = data['name']
        pbar.set_description(f"Solving {name}")
        
        # 1. New process per ring
        model = LevinWenSolver(data['N'].to(DEVICE)).to(DEVICE)
        
        # 2. Optimize
        opt = optim.LBFGS(model.parameters(), lr=1, max_iter=20, line_search_fn="strong_wolfe")
        
        def closure():
            opt.zero_grad()
            l = model.loss()
            l.backward()
            return l

        try:
            for _ in range(15): opt.step(closure)
        except:
            pass 

        # 3. Check Success
        final_loss = model.loss().item()
        
        if final_loss < 1e-5:
            model.save(name)
            solved_count += 1
    
    print(f"\n\n--- COMPLETED ---")
    print(f"Successfully Solved: {solved_count} / {len(dataset)}")
    print(f"Solutions saved to: {SAVE_DIR}/")

if __name__ == "__main__":
    main()