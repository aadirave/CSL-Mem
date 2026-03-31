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

    q_15 = torch.quantile(csl_epoch_15, torch.tensor([0.25, 0.5, 0.75]), dim=0)
    q_20 = torch.quantile(csl_end, torch.tensor([0.25, 0.5, 0.75]), dim=0)

    # partition into 4 quartiles; 4th is most memorized/highest CSL
    def get_quartiles(values, quantiles):
        buckets = torch.zeros_like(values, dtype=torch.long)
        buckets[values > quantiles[0]] = 1
        buckets[values > quantiles[1]] = 2
        buckets[values > quantiles[2]] = 3
        return buckets

    buckets_15 = get_quartiles(csl_epoch_15, q_15)
    buckets_20 = get_quartiles(csl_end, q_20)

    # rows (i) = epoch 15 bucket, cols (j) = epoch 100 bucket
    transition_matrix = torch.zeros((4, 4), dtype=torch.long)
    for i in range(4):
        for j in range(4):
            transition_matrix[i, j] = torch.sum((buckets_15 == i) & (buckets_20 == j))

    print("\n--- CSL Quartile Transition Matrix (Epoch 15 -> Epoch 100) ---")
    print("Rows: Epoch 15 Quartile (Q1=0 to Q4=3)")
    print("Cols: Epoch 100 Quartile (Q1=0 to Q4=3)")
    print("Q1 represents lowest CSL (easiest), Q4 represents highest CSL (memorized)\n")

    print(f"{'':>6} | {'Q1 (0)':>8} | {'Q2 (1)':>8} | {'Q3 (2)':>8} | {'Q4 (3)':>8} | {'Total':>8}")
    print("-" * 65)

    for i in range(4):
        row_str = f"Q{i+1} ({i}) | "
        row_total = 0
        for j in range(4):
            count = transition_matrix[i, j].item()
            row_total += count
            row_str += f"{count:>8} | "
        
        row_str += f"{row_total:>8}"
        print(row_str)

    print("-" * 65)

    same_bucket_count = torch.sum(torch.diagonal(transition_matrix)).item()
    changed_bucket_count = num_samples - same_bucket_count
    
    print(f"\nSummary:")
    print(f"Samples staying in the same quartile: {same_bucket_count} ({same_bucket_count/num_samples*100:.2f}%)")
    print(f"Samples changing quartiles:           {changed_bucket_count} ({changed_bucket_count/num_samples*100:.2f}%)")

    npz_file = args.npz_file
    if os.path.exists(npz_file):
        print(f"\nLoading memorization scores from {npz_file}...")
        npz_data = np.load(npz_file)
        if 'tr_mem' in npz_data:
            tr_mem = torch.tensor(npz_data['tr_mem'], device=device)
            print(f"Loaded 'tr_mem' with shape {tr_mem.shape}")
            
            # Use configurable threshold
            mem_threshold = args.mem_threshold
            npz_memorized = (tr_mem >= mem_threshold)
            
            # CSL Memorized is considered Q4 (bucket 3)
            csl_memorized = (buckets_20 == 3)
            
            npz_mem_count = npz_memorized.sum().item()
            csl_mem_count = csl_memorized.sum().item()
            intersection = (npz_memorized & csl_memorized).sum().item()
            
            precision = intersection / csl_mem_count if csl_mem_count > 0 else 0
            recall = intersection / npz_mem_count if npz_mem_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n--- Similarity with NPZ Memorization Score (Threshold: {mem_threshold}) ---")
            print(f"Total NPZ Memorized (tr_mem >= {mem_threshold}): {npz_mem_count} / {num_samples}")
            print(f"Total CSL Memorized (Q4): {csl_mem_count} / {num_samples}")
            print(f"Intersection: {intersection}")
            print(f"Precision (CSL predicting NPZ): {precision:.4f}")
            print(f"Recall (CSL capturing NPZ): {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # Distribution of NPZ memorized samples across CSL quartiles
            print("\nDistribution of NPZ Memorized Samples across CSL Epoch 100 Quartiles:")
            print(f"{'Quartile':>8} | {'Count':>8} | {'% of Q':>10}")
            print("-" * 34)
            for i in range(4):
                q_mask = (buckets_20 == i)
                q_size = q_mask.sum().item()
                q_npz_mem = (q_mask & npz_memorized).sum().item()
                q_pct = (q_npz_mem / q_size * 100) if q_size > 0 else 0
                print(f"Q{i+1:<7} | {q_npz_mem:>8} | {q_pct:>9.2f}%")
        else:
            print("Warning: 'tr_mem' not found in the npz file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CSL Quartiles and compare with NPZ tracking")
    parser.add_argument("--mem_threshold", type=float, default=0.8, help="Threshold for a tr_mem score to be considered memorized")
    parser.add_argument("--npz_file", type=str, default="./matrices/cifar100_infl_matrix.npz", help="Path to the npz file containing memorization scores")
    args = parser.parse_args()
    main(args)
