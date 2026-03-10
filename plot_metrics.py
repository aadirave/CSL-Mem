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

    # non-normalized
    cumulative_loss_tracker = torch.cumsum(loss_tracker, dim=0)
    cumulative_grad_norm_tracker = torch.cumsum(grad_norm_tracker, dim=0)

    # Calculate CSG: Sum of squared input gradients over all epochs
    # CSG(z_i) ≈ sum_{t=0}^{Tmax} ||∇x_i ℓ(w_t)||_2^2
    print("Calculating CSG scores...")
    csg_scores = (grad_norm_tracker**2).sum(dim=0)

    print("Calculating CSL scores...")
    csl_scores = loss_tracker.sum(dim=0)

    # Identify top-k, bottom-k CSG samples
    k = 10
    sorted_csg, sorted_indices = torch.sort(csg_scores)

    lowest_csg_indices = sorted_indices[:k]
    highest_csg_indices = sorted_indices[-k:]
    
    mid_idx = num_samples // 2

    # Identify median-k CSL samples
    sorted_csl, sorted_csl_indices = torch.sort(csl_scores)
    median_csl_indices = sorted_csl_indices[mid_idx - k//2 : mid_idx + k//2]

    print(f"\nLowest CSG samples (indices): {lowest_csg_indices.tolist()}")
    print(f"CSG scores: {sorted_csg[:k].tolist()}")

    print(f"\nHighest CSG samples (indices): {highest_csg_indices.tolist()}")
    print(f"CSG scores: {sorted_csg[-k:].tolist()}")

    print(f"Median CSL samples (indices): {median_csl_indices.tolist()}")

    os.makedirs("./plots", exist_ok=True)

    # plot Loss Trajectories
    plt.figure(figsize=(10, 6))

    # plot lowest CSG (easy)
    for idx in lowest_csg_indices:
        plt.plot(
            cumulative_loss_tracker[:, idx].numpy(),
            linestyle="-",
            alpha=0.7,
            label=f"Low CSG (idx {idx})",
        )

    # plot highest CSG (memorized)
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("./plots/loss_trajectories.png", dpi=300)
    plt.close()

    # plot gradient norm trajectories
    plt.figure(figsize=(10, 6))

    # plot lowest CSG
    for idx in lowest_csg_indices:
        plt.plot(
            cumulative_grad_norm_tracker[:, idx].numpy(),
            linestyle="-",
            alpha=0.7,
            label=f"Low CSG (idx {idx})",
        )

    # plot highest CSG
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

    # ------------------ Plot Median Trajectories ------------------
    # Median CSL Loss Trajectories
    plt.figure(figsize=(10, 6))
    for idx in median_csl_indices:
        plt.plot(cumulative_loss_tracker[:, idx].numpy(), linestyle="-", alpha=0.7, label=f"Median CSL (idx {idx})")
    plt.title("Cumulative Loss Trajectories (Median CSL)")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Loss (log scale)")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("./plots/median_csl_loss_trajectories.png", dpi=300)
    plt.close()

    # Median CSL Grad Norm Trajectories
    plt.figure(figsize=(10, 6))
    for idx in median_csl_indices:
        plt.plot(cumulative_grad_norm_tracker[:, idx].numpy(), linestyle="-", alpha=0.7, label=f"Median CSL (idx {idx})")
    plt.title("Cumulative Gradient Norm Trajectories (Median CSL)")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Gradient Norm (log scale)")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("./plots/median_csl_grad_norm_trajectories.png", dpi=300)
    plt.close()

    print(
        "\nSaved plots to ./plots/loss_trajectories.png and ./plots/grad_norm_trajectories.png"
    )
    print("Also saved median trajectory plots to ./plots/")


if __name__ == "__main__":
    main()
