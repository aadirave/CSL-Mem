import torch
import matplotlib.pyplot as plt
import os


def main():
    device = torch.device("cpu")
    tracking_file = "./logs/tracking_data.pt"

    if not os.path.exists(tracking_file):
        print(f"Error: {tracking_file} not found.")
        return

    data = torch.load(tracking_file, map_location=device)
    loss_tracker = data["loss_tracker"]
    grad_norm_tracker = data["grad_norm_tracker"]
    
    num_epochs, num_samples = loss_tracker.shape
    print(f"Loaded tracking data for {num_samples} samples over {num_epochs} epochs.")

    epoch_norm = torch.arange(1, num_epochs + 1, dtype=torch.float32).unsqueeze(1)

    cumulative_loss_tracker = torch.cumsum(loss_tracker, dim=0) / epoch_norm
    cumulative_grad_norm_tracker = torch.cumsum(grad_norm_tracker, dim=0) / epoch_norm

    # Calculate CSG: Sum of squared input gradients over all epochs
    # CSG(z_i) ≈ sum_{t=0}^{Tmax} ||∇x_i ℓ(w_t)||_2^2
    print("Calculating CSG scores...")
    csg_scores = (grad_norm_tracker**2).sum(dim=0)

    # Identify top-k highest and bottom-k lowest CSG samples
    k = 10
    sorted_csg, sorted_indices = torch.sort(csg_scores)

    lowest_csg_indices = sorted_indices[:k]
    highest_csg_indices = sorted_indices[-k:]

    print(f"\nLowest CSG samples (indices): {lowest_csg_indices.tolist()}")
    print(f"CSG scores: {sorted_csg[:k].tolist()}")

    print(f"\nHighest CSG samples (indices): {highest_csg_indices.tolist()}")
    print(f"CSG scores: {sorted_csg[-k:].tolist()}")

    # Ensure plots directory exists
    os.makedirs("./plots", exist_ok=True)

    # 1. Plot Loss Trajectories
    plt.figure(figsize=(10, 6))

    # Plot lowest CSG (easy examples)
    for idx in lowest_csg_indices:
        plt.plot(
            cumulative_loss_tracker[:, idx].numpy(),
            linestyle="-",
            alpha=0.7,
            label=f"Low CSG (idx {idx})",
        )

    # Plot highest CSG (hard/memorized examples)
    for idx in highest_csg_indices:
        plt.plot(
            cumulative_loss_tracker[:, idx].numpy(),
            linestyle="--",
            alpha=0.7,
            label=f"High CSG (idx {idx})",
        )

    plt.title("Cumulative Loss Trajectories")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Loss (log scale)")
    plt.yscale("log")
    # Put legend outside to avoid clutter
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("./plots/loss_trajectories.png", dpi=300)
    plt.close()

    # 2. Plot Gradient Norm Trajectories
    plt.figure(figsize=(10, 6))

    # Plot lowest CSG
    for idx in lowest_csg_indices:
        plt.plot(
            cumulative_grad_norm_tracker[:, idx].numpy(),
            linestyle="-",
            alpha=0.7,
            label=f"Low CSG (idx {idx})",
        )

    # Plot highest CSG
    for idx in highest_csg_indices:
        plt.plot(
            cumulative_grad_norm_tracker[:, idx].numpy(),
            linestyle="--",
            alpha=0.7,
            label=f"High CSG (idx {idx})",
        )

    plt.title("Cumulative Gradient Norm Trajectories")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Gradient Norm (log scale)")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("./plots/grad_norm_trajectories.png", dpi=300)
    plt.close()

    print(
        "\nSaved plots to ./plots/loss_trajectories.png and ./plots/grad_norm_trajectories.png"
    )


if __name__ == "__main__":
    main()
