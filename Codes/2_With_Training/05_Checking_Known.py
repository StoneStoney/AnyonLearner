import torch
import numpy as np
import os
import glob

DATA_DIR = "potential_categories"
OUTPUT_FILE = "physics_analysis_report.txt"

def calculate_d(N):
    try:
        M = torch.sum(N, dim=0).cpu().numpy()
        vals, vecs = np.linalg.eig(M)
        max_idx = np.argmax(np.abs(vals))
        d = np.abs(vecs[:, max_idx])
        return d / d[0]
    except: return None

def analyze_physics():
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    print(f"Scanning {len(files)} files for known physical phases...\n")
    
    with open(OUTPUT_FILE, "w") as f_out:
        header = f"Scanning {len(files)} files for known physical phases...\n\n"
        f_out.write(header)

        found_phases = {
            "Toric Code": 0,
            "Non-Abelian": 0,
            "Trivial/Broken": 0
        }

        header_row = f"{'FILENAME':<35} | {'D_VEC':<35} | {'ID'}"
        print(header_row)
        print("-" * 90)
        f_out.write(header_row + "\n")
        f_out.write("-" * 90 + "\n")

        for i, f in enumerate(files):
            try:
                data = torch.load(f, map_location='cpu', weights_only=False)
                N = data['N']
                d = calculate_d(N)
                
                d_clean = np.round(d, 3)
                
                if len(d) == 4 and np.allclose(d, [1., 1., 1., 1.], atol=0.01):
                    phase = "Toric Code / Z4"
                    found_phases["Toric Code"] += 1

                elif np.any(d > 1.01):
                    phase = "Non-Abelian"
                    found_phases["Non-Abelian"] += 1
                    
                else:
                    phase = "Unknown/Trivial"
                    found_phases["Trivial/Broken"] += 1

                line = f"{os.path.basename(f):<35} | {str(d_clean):<35} | {phase}"
                f_out.write(line + "\n")
                
                if i < 30:
                    print(line)

            except: pass

        separator = "-" * 90
        print(separator)
        f_out.write(separator + "\n")
        
        print("\nSUMMARY ")
        f_out.write("\nSUMMARY \n")
        
        for k, v in found_phases.items():
            line = f"{k}: {v} found"
            print(line)
            f_out.write(line + "\n")
            
    print(f"\n Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_physics()