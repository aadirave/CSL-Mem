import torch
import os

def main():
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


if __name__ == "__main__":
    main()
