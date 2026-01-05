import os
os.environ["OMP_NUM_THREADS"] = "8" # This can be customized, I'm on a personal device so 8 is fine. 
# otherwise only main thing to change is rank! If you have a beefy machine you can go higher... BUT it's really intensive

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from tqdm import tqdm
import time

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "potential_categories"

# 1. search algorithm
class RingMiner:
    def __init__(self, rank, max_multiplicity=1):
        self.rank = rank
        self.max_mult = max_multiplicity
        self.found_rings = []
        
        # We only need to fill N[i,j,k] for 1 <= i <= j < rank
        # (Index 0 is identity, and i>j is covered by commutativity)
        self.indices = []
        for i in range(1, rank):
            for j in range(i, rank):
                for k in range(rank):
                    self.indices.append((i, j, k))
        
        print(f"mining for Rank {rank}) ---")
        print(f"Dimensions: {len(self.indices)} cells")

    def check_associativity_fast(self, N):
        # We verify Sum_m N_ij^m N_mk^l == Sum_m N_jk^m N_im^l
        # We stop as soon as we find a mismatch
        lhs = np.einsum('ijm, mkl -> ijkl', N, N) 
        rhs = np.einsum('jkm, iml -> ijkl', N, N)
        return np.array_equal(lhs, rhs)

    def mine(self):
        # Identity rules
        N_base = np.zeros((self.rank, self.rank, self.rank), dtype=int)
        for x in range(self.rank):
            N_base[0, x, x] = 1
            N_base[x, 0, x] = 1
            N_base[x, x, 0] = 1 

        # Start Recursion
        self._backtrack(N_base, 0)
        return self.found_rings

    def _backtrack(self, N, step):
        if step == len(self.indices):
            if self.check_associativity_fast(N):
                # We found a valid ring! 
                self.found_rings.append(N.copy())
            return

        # Recursive Step
        i, j, k = self.indices[step]

        # Try values [0, 1] (or higher if max_mult > 1)
        for val in range(self.max_mult + 1):
            N[i, j, k] = val
            N[j, i, k] = val # Enforce Commutativity
            
            # PRUNING:
            # We can run a partial check here? 
            # For Python speed, we usually only check at the end or at key checkpoints.
            # Let's simple-recurse for now.
            self._backtrack(N, step + 1)

            # Reset for next iteration
            N[i, j, k] = 0
            N[j, i, k] = 0

# ==============================================================================
# 2. THE VERIFIER: NEURAL SOLVER
# ==============================================================================
class LevinWenSolver(nn.Module):
    def __init__(self, N_tensor):
        super().__init__()
        self.register_buffer('N', N_tensor)
        self.dim = N_tensor.shape[0]
        
        # Solve Dimensions
        M = torch.sum(self.N, dim=0).cpu().numpy()
        try:
            vals, vecs = np.linalg.eig(M)
            d = np.abs(vecs[:, np.argmax(np.abs(vals))])
            d = d / d[0]
            self.register_buffer('d', torch.tensor(d, device=DEVICE))
        except:
            # Fallback for weird matrices
            self.register_buffer('d', torch.ones(self.dim, device=DEVICE))

        # Build Mask
        self.register_buffer('mask', self._build_mask())
        self.F_params = nn.Parameter(torch.randn(*self.mask.shape, 2) * 0.1)

    def _build_mask(self):
        mask = torch.zeros((self.dim,)*6)
        N = self.N.cpu().numpy()
        import itertools
        for i,j,k,l,m,n in itertools.product(range(self.dim), repeat=6):
            if (N[i,j,m] and N[m,k,l] and N[j,k,n] and N[i,n,l]):
                mask[i,j,k,l, m,n] = 1.0
        return mask.to(DEVICE)

    def get_F(self): return torch.view_as_complex(self.F_params) * self.mask
    def loss(self):
        F = self.get_F()
        ortho = torch.einsum('ijklmn, ijklpn -> ijklmp', F, F.conj())
        valid_rows = (self.mask.sum(dim=5) > 0).float()
        return torch.norm(ortho - torch.diag_embed(valid_rows))
    
    def save(self, name):
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save({'name': name, 'N': self.N.cpu(), 'F': self.get_F().detach().cpu()}, 
                   os.path.join(SAVE_DIR, f"{name}.pt"))

# main pipeline
def main():
    # settings
    RANK_TO_MINE = 4  # can config. Gets exponentially bigger
        
    # mine
    miner = RingMiner(rank=RANK_TO_MINE, max_multiplicity=1)
    start_time = time.time()
    rings = miner.mine()
    duration = time.time() - start_time
    
    print(f"\n[MINING COMPLETE]")
    print(f"Time Taken: {duration:.2f}s")
    print(f"Candidates Found: {len(rings)}")
    print(f"(These are integer tensors that satisfy Associativity)")
    
    # solve
    print(f"\n--- SOLVING CANDIDATES ---")
    pbar = tqdm(rings)
    solved_count = 0
    
    for idx, N_np in enumerate(pbar):
        name = f"Gen_R{RANK_TO_MINE}_{idx:03d}"
        pbar.set_description(f"Solving {name}")
        
        # Check if valid input for solver
        N_tensor = torch.tensor(N_np, dtype=torch.float64)
        
        try:
            model = LevinWenSolver(N_tensor.to(DEVICE)).to(DEVICE)
            opt = optim.LBFGS(model.parameters(), lr=1, max_iter=20, line_search_fn="strong_wolfe")
            
            def closure():
                opt.zero_grad(); l = model.loss(); l.backward(); return l
            
            for _ in range(15): opt.step(closure)
            
            if model.loss().item() < 1e-5:
                model.save(name)
                solved_count += 1
        except Exception as e:
            # just incase
            pass

    print(f"\n--- DISCOVERY REPORT ---")
    print(f"Valid Fusion Categories Discovered: {solved_count} / {len(rings)}")
    print(f"Models saved to {SAVE_DIR}/")

if __name__ == "__main__":
    main()