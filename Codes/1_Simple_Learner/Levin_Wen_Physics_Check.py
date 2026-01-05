import os
import glob
import pandas as pd
import torch
import sys

torch.set_default_dtype(torch.float64)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "levin_wen_solutions")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "levin_wen_physics_report.csv")

class HamiltonianAnalyzer:
    def __init__(self, solution_path):
        # loading with CPU
        self.data = torch.load(solution_path, map_location=torch.device('cpu'))
        self.name = self.data['name']
        self.F = self.data['F']
        self.d = self.data['d']
        self.D_total = self.data['D']

    def check_physics(self):
        results = {}
        
        # 1. HERMITICITY CHECK
        # F_{mn} (F_{pn})* = delta_{mp}
        ortho = torch.einsum('ijklmn, ijklpn -> ijklmp', self.F, self.F.conj())
        
        diags = torch.diagonal(ortho, dim1=-2, dim2=-1)
        
        valid_mask = diags.real > 0.1
        valid_diags = diags[valid_mask]
        
        if len(valid_diags) > 0:
            # Check how far they are from 1.0 + 0.0j
            max_error = torch.max(torch.abs(valid_diags - 1.0)).item()
        else:
            max_error = 999.0 

        results['Hermitian_Error'] = max_error

        # 2. UNITARITY CHECK
        # We take the real part of d just in case it's stored as complex which might happen
        if torch.is_complex(self.d):
            min_d = torch.min(self.d.real).item()
        else:
            min_d = torch.min(self.d).item()
            
        results['Min_Quantum_Dim'] = min_d
        results['Is_Unitary'] = (min_d > 0) and (max_error < 1e-3)

        max_imag = torch.max(torch.abs(self.F.imag)).item()
        results['Time_Reversal_Violation'] = max_imag
        results['Is_Chiral'] = max_imag > 1e-5

        results['Total_D'] = self.D_total.item()

        return results

# main
def main():
    print(f"Looking for models in: {SOLUTIONS_DIR}")

    if not os.path.exists(SOLUTIONS_DIR):
        print(f"ERROR: Could not find folder 'levin_wen_solutions'. Run anyon_learner first.")
        return

    files = glob.glob(os.path.join(SOLUTIONS_DIR, "*.pt"))
    print(f"Found {len(files)} solution files.")

    report = []
    print("Analyzing..", end="", flush=True)

    for i, fpath in enumerate(files):
        try:
            analyzer = HamiltonianAnalyzer(fpath)
            metrics = analyzer.check_physics()
            metrics['Category'] = analyzer.name
            report.append(metrics)
            if i % 5 == 0: print(".", end="", flush=True)
        except Exception as e:
            print(f"\nFailed on {os.path.basename(fpath)}: {e}")

    print("\nDone.")

    if len(report) > 0:
        df = pd.DataFrame(report)
        cols = ['Category', 'Is_Unitary', 'Is_Chiral', 'Min_Quantum_Dim', 'Hermitian_Error', 'Time_Reversal_Violation', 'Total_D']
        df = df[cols]
        df = df.sort_values(by='Total_D')

        print(f"\n--- Preview (Top 5) ---")
        print(df.head(5).to_string(index=False))

        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nReport saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()