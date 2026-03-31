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
    
    # Calculate the flattening metric
    # The epoch when cumulative loss reaches `flatten_threshold` of the total loss.
    target_csl = cumulative_loss_tracker[-1] * args.flatten_threshold
    
    # argmax returns the first index where the condition is true
    # adding 1 to make it 1-indexed epoch (range 1-100) instead of 0-99 index.
    flatten_epochs = (cumulative_loss_tracker >= target_csl).float().argmax(dim=0).numpy() + 1
    
    flatten_min = flatten_epochs.min()
    flatten_max = flatten_epochs.max()
    flatten_mean = flatten_epochs.mean()
    
    print(f"\nMetric: Epoch at which {args.flatten_threshold*100:.0f}% of total loss is reached (Flatten Epoch)")
    print(f"Stats -> Min: {flatten_min}, Max: {flatten_max}, Mean: {flatten_mean:.2f}")

    npz_file = args.npz_file
    if os.path.exists(npz_file):
        print(f"\nLoading memorization scores from {npz_file}...")
        npz_data = np.load(npz_file)
        if 'tr_mem' in npz_data:
            tr_mem = npz_data['tr_mem']
            
            mem_threshold = args.mem_threshold
            npz_memorized = (tr_mem >= mem_threshold).astype(int)

            # 1. Pearson/Spearman Correlation between Flatten Epoch and tr_mem score
            spearman_corr, p_value = spearmanr(flatten_epochs, tr_mem)
            print(f"\n--- Continuous Correlation ---")
            print(f"Spearman Rank Correlation (Flattening predicting Memorization): {spearman_corr:.4f} (p-value: {p_value:.4e})")

            # 2. AUC Score: how well the `flatten_epochs` predict binary `npz_memorized`
            if npz_memorized.sum() > 0 and npz_memorized.sum() < num_samples:
                auc_score = roc_auc_score(npz_memorized, flatten_epochs)
                print(f"ROC-AUC Score (Later Flattening -> Memorized): {auc_score:.4f}")
            else:
                print("Could not compute ROC-AUC score because there is only one class in npz_memorized.")
            
            # 3. Bin output into deciles for consistency
            num_bins = args.num_bins
            
            # Use unique quantiles to prevent binning error if heavily skewed
            quantiles = np.unique(np.quantile(flatten_epochs, np.linspace(0, 1, num_bins + 1)))
            num_bins_actual = len(quantiles) - 1
            
            if num_bins_actual < num_bins:
                print(f"Note: Using {num_bins_actual} bins because many samples hit the exact same flatten epoch.")
                
            buckets = np.digitize(flatten_epochs, quantiles[1:-1])
            
            top_class = (buckets == (num_bins_actual - 1))
            
            npz_mem_count = npz_memorized.sum()
            class_mem_count = top_class.sum()
            intersection = (npz_memorized & top_class).sum()
            
            precision = intersection / class_mem_count if class_mem_count > 0 else 0
            recall = intersection / npz_mem_count if npz_mem_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n--- Similarity with NPZ Memorization Score (Threshold: {mem_threshold}) ---")
            print(f"Total NPZ Memorized (tr_mem >= {mem_threshold}): {npz_mem_count} / {num_samples}")
            print(f"Total Memorized by Metric (Highest Flatten Bin B{num_bins_actual}): {class_mem_count} / {num_samples}")
            print(f"Intersection: {intersection}")
            print(f"Precision (Flatten Metric predicting NPZ): {precision:.4f}")
            print(f"Recall (Flatten Metric capturing NPZ): {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            print(f"\nDistribution of NPZ Memorized Samples across Flattening Epoch Bins:")
            print(f"B1 = Flattens Very Early (Easy) -> B{num_bins_actual} = Flattens Very Late (Memorized)")
            print(f"{'Bin':>10} | {'Range (Epoch)':>18} | {'Count':>10} | {'% of B':>12}")
            print("-" * 59)
            
            for i in range(num_bins_actual):
                q_mask = (buckets == i)
                q_size = q_mask.sum()
                q_npz_mem = (q_mask & npz_memorized).sum()
                q_pct = (q_npz_mem / q_size * 100) if q_size > 0 else 0
                
                # Format range string
                if i == 0:
                    range_str = f"<= {quantiles[1]:.0f}"
                elif i == num_bins_actual - 1:
                    range_str = f"> {quantiles[-2]:.0f}"
                else:
                    range_str = f"{quantiles[i]:.0f} to {quantiles[i+1]:.0f}"
                    
                print(f"B{i+1:<9} | {range_str:>18} | {q_size:>10} | {q_pct:>11.2f}%")
        else:
            print("Warning: 'tr_mem' not found in the npz file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze when loss flattens out and compare with NPZ tracking")
    parser.add_argument("--flatten_threshold", type=float, default=0.90, help="Fraction of total cumulative loss to consider 'flattened'")
    parser.add_argument("--mem_threshold", type=float, default=0.8, help="Threshold for a tr_mem score to be considered memorized")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins to sort the flattening epochs into")
    parser.add_argument("--npz_file", type=str, default="./matrices/cifar100_infl_matrix.npz", help="Path to the npz file containing memorization scores")
    args = parser.parse_args()
    main(args)
