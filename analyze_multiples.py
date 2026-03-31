import torch
import os
import argparse
import numpy as np

def main(args):
    device = torch.device("cpu")
    tracking_file = "./logs/tracking_data.pt"

    if not os.path.exists(tracking_file):
        print(f"Error: {tracking_file} not found.")
        return

    print(f"Loading {tracking_file}...")
    data = torch.load(tracking_file, map_location=device)
    loss_tracker = data["loss_tracker"]

    num_epochs, num_samples = loss_tracker.shape
    cumulative_loss_tracker = torch.cumsum(loss_tracker, dim=0)

    if args.epoch < 1 or args.epoch > num_epochs:
        print(f"Error: Please specify an epoch between 1 and {num_epochs}")
        return

    csl_target = cumulative_loss_tracker[args.epoch - 1]

    # Calculate 5th percentile
    val_5pct = torch.quantile(csl_target, 0.05).item()
    max_val = csl_target.max().item()
    
    print(f"Epoch {args.epoch} CSL 5th Percentile: {val_5pct:.4f}")
    print(f"Epoch {args.epoch} CSL Max Value:      {max_val:.4f}")

    # Create boundaries: [val_5pct, 2*val_5pct, 3*val_5pct...]
    boundaries_list = []
    mult = 1
    while mult * val_5pct < max_val:
        boundaries_list.append(mult * val_5pct)
        mult += 1
        
    boundaries = torch.tensor(boundaries_list, device=device)
    num_bins = len(boundaries) + 1
    print(f"Total number of bins created: {num_bins} (Max multiplier: {mult-1}x)")

    # Bucketize
    # Bucket 0: < val_5pct
    # Bucket 1: val_5pct to 2*val_5pct, etc.
    buckets = torch.bucketize(csl_target, boundaries)

    npz_file = args.npz_file
    if os.path.exists(npz_file):
        print(f"\nLoading memorization scores from {npz_file}...")
        npz_data = np.load(npz_file)
        if 'tr_mem' in npz_data:
            tr_mem = torch.tensor(npz_data['tr_mem'], device=device)
            
            mem_threshold = args.mem_threshold
            npz_memorized = (tr_mem >= mem_threshold)
            npz_mem_count = npz_memorized.sum().item()
            
            print(f"\n--- Prediction of NPZ Memorization (tr_mem >= {mem_threshold}) ---")
            print(f"Distribution of samples across Multiple Bins at Epoch {args.epoch}:")
            print(f"{'Bin':>10} | {'Range (CSL)':>20} | {'Total Samples':>15} | {'NPZ Memorized':>15} | {'% Memorized in Bin':>20}")
            print("-" * 92)
            
            for i in range(num_bins):
                q_mask = (buckets == i)
                q_size = q_mask.sum().item()
                if q_size == 0:
                    continue
                q_npz_mem = (q_mask & npz_memorized).sum().item()
                q_pct = (q_npz_mem / q_size * 100) if q_size > 0 else 0
                
                # Format range string
                if i == 0:
                    range_str = f"< {val_5pct:.2f}"
                    bin_name = "1 (<1x)"
                elif i == num_bins - 1:
                    range_str = f">= {boundaries_list[-1]:.2f}"
                    bin_name = f"{i+1} (>={i}x)"
                else:
                    range_str = f"{boundaries_list[i-1]:.2f} - {boundaries_list[i]:.2f}"
                    bin_name = f"{i+1} ({i}x - {i+1}x)"
                    
                print(f"{bin_name:>10} | {range_str:>20} | {q_size:>15} | {q_npz_mem:>15} | {q_pct:>19.2f}%")
        else:
            print("Warning: 'tr_mem' not found in the npz file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CSL by 5th percentile multiples")
    parser.add_argument("--epoch", type=int, default=100, help="Epoch to analyze the CSL for (e.g. 15 or 100)")
    parser.add_argument("--mem_threshold", type=float, default=0.8, help="Threshold for a tr_mem score to be considered memorized")
    parser.add_argument("--npz_file", type=str, default="./matrices/cifar100_infl_matrix.npz", help="Path to the npz file containing memorization scores")
    args = parser.parse_args()
    main(args)
