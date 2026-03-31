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
    print(f"Loaded tracking data for {num_samples} samples over {num_epochs} epochs.")

    cumulative_loss_tracker = torch.cumsum(loss_tracker, dim=0)

    csl_epoch_15 = cumulative_loss_tracker[14]
    csl_end = cumulative_loss_tracker[-1]

    num_bins = args.num_bins
    quant_levels = torch.linspace(0, 1, num_bins + 1)[1:-1].to(device)
    
    q_15 = torch.quantile(csl_epoch_15, quant_levels, dim=0)
    q_20 = torch.quantile(csl_end, quant_levels, dim=0)

    buckets_15 = torch.bucketize(csl_epoch_15, q_15)
    buckets_20 = torch.bucketize(csl_end, q_20)

    # rows (i) = epoch 15 bucket, cols (j) = epoch 100 bucket
    transition_matrix = torch.zeros((num_bins, num_bins), dtype=torch.long)
    for i in range(num_bins):
        for j in range(num_bins):
            transition_matrix[i, j] = torch.sum((buckets_15 == i) & (buckets_20 == j))

    print(f"\n--- CSL Bin Transition Matrix (Epoch 15 -> Epoch 100, {num_bins} Bins) ---")
    print(f"Rows: Epoch 15 Bin (B1=0 to B{num_bins}={num_bins-1})")
    print(f"Cols: Epoch 100 Bin (B1=0 to B{num_bins}={num_bins-1})")
    print(f"B1 represents lowest CSL (easiest), B{num_bins} represents highest CSL (memorized)\n")

    # Header
    header = f"{'':>6} | " + " | ".join([f"B{i+1:<2} ({i:<1})" for i in range(num_bins)]) + " |    Total"
    print(header)
    print("-" * len(header))

    for i in range(num_bins):
        row_str = f"B{i+1:<2}({i:<1}) | "
        row_total = 0
        for j in range(num_bins):
            count = transition_matrix[i, j].item()
            row_total += count
            row_str += f"{count:>8} | "
        
        row_str += f"{row_total:>8}"
        print(row_str)

    print("-" * len(header))

    same_bucket_count = torch.sum(torch.diagonal(transition_matrix)).item()
    changed_bucket_count = num_samples - same_bucket_count
    
    print(f"\nSummary:")
    print(f"Samples staying in the same bin: {same_bucket_count} ({same_bucket_count/num_samples*100:.2f}%)")
    print(f"Samples changing bins:           {changed_bucket_count} ({changed_bucket_count/num_samples*100:.2f}%)")

    # Calculate Median CSL ratio
    median_b0 = torch.median(csl_end[buckets_20 == 0]).item()
    median_b_last = torch.median(csl_end[buckets_20 == (num_bins - 1)]).item()
    print(f"\nMedian CSL for Lowest Bin (B1): {median_b0:.4f}")
    print(f"Median CSL for Highest Bin (B{num_bins}): {median_b_last:.4f}")
    csl_ratio = median_b_last / median_b0 if median_b0 != 0 else float('inf')
    print(f"CSL Ratio (Highest / Lowest): {csl_ratio:.4f}")


    npz_file = args.npz_file
    if os.path.exists(npz_file):
        print(f"\nLoading memorization scores from {npz_file}...")
        npz_data = np.load(npz_file)
        if 'tr_mem' in npz_data:
            tr_mem = torch.tensor(npz_data['tr_mem'], device=device)
            print(f"Loaded 'tr_mem' with shape {tr_mem.shape}")
            
            mem_threshold = args.mem_threshold
            npz_memorized = (tr_mem >= mem_threshold)
            
            csl_memorized = (buckets_20 == (num_bins - 1))
            
            npz_mem_count = npz_memorized.sum().item()
            csl_mem_count = csl_memorized.sum().item()
            intersection = (npz_memorized & csl_memorized).sum().item()
            
            precision = intersection / csl_mem_count if csl_mem_count > 0 else 0
            recall = intersection / npz_mem_count if npz_mem_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n--- Similarity with NPZ Memorization Score (Threshold: {mem_threshold}) ---")
            print(f"Total NPZ Memorized (tr_mem >= {mem_threshold}): {npz_mem_count} / {num_samples}")
            print(f"Total CSL Memorized (B{num_bins}): {csl_mem_count} / {num_samples}")
            print(f"Intersection: {intersection}")
            print(f"Precision (CSL predicting NPZ): {precision:.4f}")
            print(f"Recall (CSL capturing NPZ): {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            print(f"\nDistribution of NPZ Memorized Samples across CSL Epoch 100 Bins:")
            print(f"{'Bin':>8} | {'Count':>8} | {'% of B':>10}")
            print("-" * 34)
            for i in range(num_bins):
                q_mask = (buckets_20 == i)
                q_size = q_mask.sum().item()
                q_npz_mem = (q_mask & npz_memorized).sum().item()
                q_pct = (q_npz_mem / q_size * 100) if q_size > 0 else 0
                print(f"B{i+1:<7} | {q_npz_mem:>8} | {q_pct:>9.2f}%")
        else:
            print("Warning: 'tr_mem' not found in the npz file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CSL Quartiles and compare with NPZ tracking")
    parser.add_argument("--mem_threshold", type=float, default=0.8, help="Threshold for a tr_mem score to be considered memorized")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of quantile bins (e.g., 4 for quartiles, 10 for deciles)")
    parser.add_argument("--npz_file", type=str, default="./matrices/cifar100_infl_matrix.npz", help="Path to the npz file containing memorization scores")
    args = parser.parse_args()
    main(args)
