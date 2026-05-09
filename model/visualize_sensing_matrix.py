import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
from datetime import datetime

def load_fpi_data(directory):
    files = glob.glob(os.path.join(directory, '*nm.txt')) + \
            glob.glob(os.path.join(directory, '*nm_*.txt'))
    
    if not files:
        raise FileNotFoundError(f"No *nm.txt or *nm_*.txt files found in {directory}")
    
    def extract_number(f):
        basename = os.path.basename(f)
        num_str = basename.split('nm')[0]
        return int(num_str)
    
    files = sorted(files, key=extract_number)
    
    def detect_skiprows(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    try:
                        parts = stripped.split('\t')
                        float(parts[0])
                        return i
                    except (ValueError, IndexError):
                        continue
        return 1
    
    skiprows = detect_skiprows(files[0])
    data = np.loadtxt(files[0], delimiter='\t', skiprows=skiprows)
    wavelengths = data[:, 0]
    n_wavelengths = len(wavelengths)
    n_states = len(files)
    
    trans_col = 1 if data.shape[1] == 2 else 2
    
    sensing_matrix = np.zeros((n_states, n_wavelengths))
    
    for i, filepath in enumerate(files):
        skiprows = detect_skiprows(filepath)
        data = np.loadtxt(filepath, delimiter='\t', skiprows=skiprows)
        sensing_matrix[i, :] = data[:, trans_col]
    
    return wavelengths, sensing_matrix

def visualize_matrix(wavelengths, sensing_matrix, output_path=None):
    N_states, M_wl = sensing_matrix.shape
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = axes[0].imshow(sensing_matrix, aspect='auto', cmap='viridis',
                         extent=[wavelengths[0], wavelengths[-1], N_states, 1])
    axes[0].set_title('Sensing Matrix $\Phi$ (T($\lambda$, d))')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('FPI State (Cavity Length Index)')
    fig.colorbar(im1, ax=axes[0])
    
    valid_cols = ~np.all(sensing_matrix == 0, axis=0)
    valid_wavelengths = wavelengths[valid_cols]
    corr_matrix = np.corrcoef(sensing_matrix[:, valid_cols], rowvar=False)
    im2 = axes[1].imshow(corr_matrix, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1,
                         extent=[valid_wavelengths[0], valid_wavelengths[-1], valid_wavelengths[-1], valid_wavelengths[0]])
    axes[1].set_title('Wavelength Cross-Correlation')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Wavelength (nm)')
    fig.colorbar(im2, ax=axes[1])
    
    U, S, V = np.linalg.svd(sensing_matrix, full_matrices=False)
    axes[2].plot(np.log10(S / S[0] + 1e-12), 'b.-')
    axes[2].set_title('Singular Value Decay (Log Scale)')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Log10(S_i / S_0)')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize FPI sensing matrix from transmission data')
    parser.add_argument('directory', nargs='?', 
                        default=r'E:\AA_repository\OneDrive - Unispectral Qingdao Microelectronics Co. LTD\01_研发\01-开发相关\07_MEMS\03_Coating\02-Macleod-仿真数据库\analysis_output\FPI-Performance\20260509_134338_202604171031-U450-MEMS-Metal-Coating-Ag-J08\data-BPF-BowEffect',
                        help='Directory containing *nm.txt files')
    args = parser.parse_args()
    
    print(f"Loading data from: {args.directory}")
    wavelengths, sensing_matrix = load_fpi_data(args.directory)
    print(f"Loaded sensing matrix: {sensing_matrix.shape[0]} states x {sensing_matrix.shape[1]} wavelengths")
    print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(os.path.dirname(script_dir), 'result')
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_basename = os.path.basename(args.directory.rstrip(os.sep))
    output_filename = f"{timestamp}_{dir_basename}.png"
    output_path = os.path.join(result_dir, output_filename)
    
    visualize_matrix(wavelengths, sensing_matrix, output_path)

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear') 
    main()
