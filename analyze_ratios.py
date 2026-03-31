import torch
import os
import argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

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

    if args.epoch_start < 1 or args.epoch_start > num_epochs or args.epoch_end < 1 or args.epoch_end > num_epochs:
        print(f"Error: Please specify epochs between 1 and {num_epochs}")
        return

    # Adding small epsilon to avoid divide by zero if start epoch loss was absolutely 0
    csl_start = cumulative_loss_tracker[args.epoch_start - 1] + 1e-8
    csl_end = cumulative_loss_tracker[args.epoch_end - 1]

    # Calculate Continuous Ratios
    ratios = (csl_end / csl_start).numpy()

    npz_file = args.npz_file
    if os.path.exists(npz_file):
        print(f"\nLoading memorization scores from {npz_file}...")
        npz_data = np.load(npz_file)
        if 'tr_mem' in npz_data:
            tr_mem = npz_data['tr_mem']
            print(f"Loaded 'tr_mem' with shape {tr_mem.shape}")
            
            mem_threshold = args.mem_threshold
            npz_memorized = (tr_mem >= mem_threshold).astype(int)

            # 1. Pearson/Spearman Correlation between Continuous Ratio and tr_mem score
            spearman_corr, p_value = spearmanr(ratios, tr_mem)
            print(f"\n--- Continuous Correlation ---")
            print(f"Spearman Rank Correlation: {spearman_corr:.4f} (p-value: {p_value:.4e})")

            # 2. AUC Score: how well the continuous `ratios` predict binary `npz_memorized`
            if npz_memorized.sum() > 0 and npz_memorized.sum() < num_samples:
                auc_score = roc_auc_score(npz_memorized, ratios)
                print(f"ROC-AUC Score (Ratio predicting Memorization): {auc_score:.4f}")
            else:
                print("Could not compute ROC-AUC score because there is only one class in npz_memorized.")
            
            # 3. Optional: Bin ratio to calculate precision/recall equivalent just to compare directly
            num_bins = args.num_bins
            ratio_quantiles = np.quantile(ratios, np.linspace(0, 1, num_bins + 1)[1:-1])
            ratio_buckets = np.digitize(ratios, ratio_quantiles)
            
            top_ratio_class = (ratio_buckets == (num_bins - 1))
            
            npz_mem_count = npz_memorized.sum()
            ratio_mem_count = top_ratio_class.sum()
            intersection = (npz_memorized & top_ratio_class).sum()
            
            precision = intersection / ratio_mem_count if ratio_mem_count > 0 else 0
            recall = intersection / npz_mem_count if npz_mem_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n--- Similarity with NPZ Memorization Score (Threshold: {mem_threshold}) ---")
            print(f"Total NPZ Memorized (tr_mem >= {mem_threshold}): {npz_mem_count} / {num_samples}")
            print(f"Total Ratio Memorized (B{num_bins}): {ratio_mem_count} / {num_samples}")
            print(f"Intersection: {intersection}")
            print(f"Precision (Ratio predicting NPZ): {precision:.4f}")
            print(f"Recall (Ratio capturing NPZ): {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            print(f"\nDistribution of NPZ Memorized Samples across Ratio Bins:")
            print(f"{'Bin':>8} | {'Count':>8} | {'% of B':>10}")
            print("-" * 34)
            for i in range(num_bins):
                q_mask = (ratio_buckets == i)
                q_size = q_mask.sum()
                q_npz_mem = (q_mask & npz_memorized).sum()
                q_pct = (q_npz_mem / q_size * 100) if q_size > 0 else 0
                print(f"B{i+1:<7} | {q_npz_mem:>8} | {q_pct:>9.2f}%")
        else:
            print("Warning: 'tr_mem' not found in the npz file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Continuous Ratios and compare with NPZ tracking")
    parser.add_argument("--epoch_start", type=int, default=15, help="Start epoch for calculating ratio")
    parser.add_argument("--epoch_end", type=int, default=100, help="End epoch for calculating ratio")
    parser.add_argument("--mem_threshold", type=float, default=0.8, help="Threshold for a tr_mem score to be considered memorized")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins to sort the Ratios into for comparison")
    parser.add_argument("--npz_file", type=str, default="./matrices/cifar100_infl_matrix.npz", help="Path to the npz file containing memorization scores")
    args = parser.parse_args()
    main(args)
