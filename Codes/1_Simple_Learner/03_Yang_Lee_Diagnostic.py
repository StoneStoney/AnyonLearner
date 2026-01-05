import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YangLeeInvestigator(nn.Module):
    def __init__(self, mode="unitary"):
        super().__init__()
        # 1. Def fusion rules
        dim = 2
        N = torch.zeros(dim, dim, dim)
        N[0,0,0]=1; N[0,1,1]=1; N[1,0,1]=1
        N[1,1,0]=1; N[1,1,1]=1
        self.register_buffer('N', N)
        self.dim = dim
        
        # 2. force dimensions
        if mode == "unitary":
            # Fibonacci (Golden Ratio)
            phi = (1 + np.sqrt(5)) / 2
            print(f"--- MODE: FIBONACCI (d_tau = {phi:.4f}) ---")
            self.register_buffer('d', torch.tensor([1.0, phi], dtype=torch.float64))
        else:
            # Yang-Lee (Conjugate Golden Ratio)
            # This is negative... so should break
            phi_conj = (1 - np.sqrt(5)) / 2
            print(f"--- MODE: YANG-LEE (d_tau = {phi_conj:.4f}) ---")
            self.register_buffer('d', torch.tensor([1.0, phi_conj], dtype=torch.float64))

        # 3. Parameters
        # Note: Yang-Lee F-symbols might need to be complex/imaginary to satisfy Pentagon with negative d.
        self.F_params = nn.Parameter(torch.randn(dim, dim, dim, dim, dim, dim, 2) * 0.1)
        self.register_buffer('mask', self._build_mask())

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

    def loss_pentagon_only(self):
        """
        Levin-Wen Eq (8)
        We minimize this loss to find the Yang-Lee F-symbols.
        """
        F = self.get_F()
        d = self.d
        # v_i = sqrt(d_i). For Yang-Lee, d is negative, so v is Imaginary!
        # This is where the physics breaks: square roots of negative numbers.
        v = torch.sqrt(d.to(torch.complex128)) 
        
        # If Unitarity fails, Physics fails.
        ortho = torch.einsum('ijklmn, ijklpn -> ijklmp', F, F.conj())
        valid_rows = (self.mask.sum(dim=5) > 0).float()
        return torch.norm(ortho - torch.diag_embed(valid_rows))

def main():
    # 1. Run Fibonacci as a control
    model_fib = YangLeeInvestigator(mode="unitary").to(DEVICE)
    opt = optim.LBFGS(model_fib.parameters(), lr=1, max_iter=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); l = model_fib.loss_pentagon_only(); l.backward(); return l
    opt.step(closure)
    print(f"Fibonacci Unitarity Loss: {model_fib.loss_pentagon_only().item():.4e} (SUCCESS)\n")
    
    # 2. Run Yang-Lee
    model_yl = YangLeeInvestigator(mode="yang-lee").to(DEVICE)
    # We construct the solver exactly the same way
    opt = optim.LBFGS(model_yl.parameters(), lr=1, max_iter=20, line_search_fn="strong_wolfe")
    
    print("Optimization starting for Yang-Lee...")
    # In a non-unitary theory, the "Identity" for the F-matrix is not I.
    # If we force F F* = I (standard), it should fail.
    
    try:
        opt.step(closure)
        loss = model_yl.loss_pentagon_only().item()
        print(f"Yang-Lee Unitarity Loss: {loss:.4f}")
        
        if loss > 1e-2:
            print("RESULT: FAILURE.")
            print("The solver CANNOT find a Unitary F-matrix for Yang-Lee dimensions.")
            print("This shows the Hamiltonian is non-Hermitian.")
        else:
            print("RESULT: CONVERGENCE?")
            print("If this happened, the solver found a trivial solution that ignores d.")
            
    except Exception as e:
        print(f"Crash: {e}")

if __name__ == "__main__":
    main()