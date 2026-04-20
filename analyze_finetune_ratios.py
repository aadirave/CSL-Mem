import torch
import os
import argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

def analyze_ratios(cumulative_loss_tracker, npz_memorized, tr_mem, num_samples, epoch_start, epoch_end, mem_threshold, num_bins, label):
    # Adding small epsilon to avoid divide by zero
    csl_start = cumulative_loss_tracker[epoch_start - 1] + 1e-8
    csl_end = cumulative_loss_tracker[epoch_end - 1]

    ratios = (csl_end / csl_start).numpy()

    spearman_corr, p_value = spearmanr(ratios, tr_mem)
    print(f"\n========== {label} (Epoch {epoch_start} to {epoch_end}) ==========")
    print(f"Spearman Rank Correlation: {spearman_corr:.4f} (p-value: {p_value:.4e})")

    if npz_memorized.sum() > 0 and npz_memorized.sum() < num_samples:
        auc_score = roc_auc_score(npz_memorized, ratios)
        print(f"ROC-AUC Score (Ratio predicting Memorization): {auc_score:.4f}")
    else:
        print("Could not compute ROC-AUC score because there is only one class in npz_memorized.")
    
    ratio_quantiles = np.quantile(ratios, np.linspace(0, 1, num_bins + 1)[1:-1])
    ratio_buckets = np.digitize(ratios, ratio_quantiles)
    
    top_ratio_class = (ratio_buckets == (num_bins - 1))
    
    npz_mem_count = npz_memorized.sum()
    ratio_mem_count = top_ratio_class.sum()
    intersection = (npz_memorized & top_ratio_class).sum()
    
    precision = intersection / ratio_mem_count if ratio_mem_count > 0 else 0
    recall = intersection / npz_mem_count if npz_mem_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Total NPZ Memorized: {npz_mem_count} / {num_samples}")
    print(f"Total Ratio Memorized (B{num_bins}): {ratio_mem_count} / {num_samples}")
    print(f"Intersection: {intersection}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

def main(args):
    device = torch.device("cpu")
    tracking_file = "./logs/tracking_data_finetune.pt"

    if not os.path.exists(tracking_file):
        print(f"Error: {tracking_file} not found. Please run train_finetune() first.")
        return

    data = torch.load(tracking_file, map_location=device)
    loss_tracker = data["loss_tracker"]

    num_epochs, num_samples = loss_tracker.shape
    cumulative_loss_tracker = torch.cumsum(loss_tracker, dim=0)

    npz_file = args.npz_file
    if not os.path.exists(npz_file):
        print(f"Error: {npz_file} not found.")
        return

    npz_data = np.load(npz_file)
    if 'tr_mem' not in npz_data:
        print("Warning: 'tr_mem' not found in the npz file.")
        return

    tr_mem = npz_data['tr_mem']
    npz_memorized = (tr_mem >= args.mem_threshold).astype(int)

    # Analyze 10 epochs before finetuning (epochs 41-50)
    analyze_ratios(cumulative_loss_tracker, npz_memorized, tr_mem, num_samples, 41, 50, args.mem_threshold, args.num_bins, "Before Finetuning")

    # Analyze 10 epochs after starting finetuning (epochs 51-60)
    analyze_ratios(cumulative_loss_tracker, npz_memorized, tr_mem, num_samples, 51, 60, args.mem_threshold, args.num_bins, "After Finetuning")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Continuous Ratios for Finetuning")
    parser.add_argument("--mem_threshold", type=float, default=0.8, help="Threshold for a tr_mem score to be considered memorized")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins")
    parser.add_argument("--npz_file", type=str, default="./matrices/cifar100_infl_matrix.npz", help="Path to npz file")
    args = parser.parse_args()
    main(args)
