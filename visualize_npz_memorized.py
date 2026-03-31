import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file", type=str, default="./matrices/cifar100_infl_matrix.npz", help="Path to the npz file containing memorization scores")
    args = parser.parse_args()

    device = torch.device("cpu")
    npz_file = args.npz_file

    if not os.path.exists(npz_file):
        print(f"Error: {npz_file} not found.")
        return

    print("Loading datasets...")
    # Load same test transforms/data without augmentation to see base images
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)), # CIFAR-100 actual mean/std
        ]
    )

    # We must use training set because indices match the tracking data
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    classes = trainset.classes

    print(f"Loading {npz_file}...")
    npz_data = np.load(npz_file)
    if 'tr_mem' not in npz_data:
        print("Error: 'tr_mem' not found in npz file.")
        return
        
    tr_mem = torch.tensor(npz_data['tr_mem'])
    
    print("Calculating Top Memorized samples according to npz...")
    k = 10
    sorted_mem, sorted_indices = torch.sort(tr_mem)
    lowest_mem_indices = sorted_indices[:k]
    highest_mem_indices = sorted_indices[-k:]

    os.makedirs("./plots", exist_ok=True)

    print("\nVisualizing Highest npz Memorized Images...")
    fig_high = plt.figure(figsize=(20, 5))
    for i, idx in enumerate(highest_mem_indices):
        img, target = trainset[idx.item()]
        # Unnormalize CIFAR-100
        img = img * torch.tensor([0.2673, 0.2564, 0.2761]).view(3, 1, 1) + torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
        ax = fig_high.add_subplot(1, k, i + 1)
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(f"Mem: {tr_mem[idx]:.3f}\nClass: {classes[target]}\nIdx: {idx}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("./plots/highest_npz_mem_images.png", dpi=300)
    plt.close()

    print("\nVisualizing Lowest npz Memorized Images...")
    fig_low = plt.figure(figsize=(20, 5))
    for i, idx in enumerate(lowest_mem_indices):
        img, target = trainset[idx.item()]
        # Unnormalize CIFAR-100
        img = img * torch.tensor([0.2673, 0.2564, 0.2761]).view(3, 1, 1) + torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
        ax = fig_low.add_subplot(1, k, i + 1)
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(f"Mem: {tr_mem[idx]:.3f}\nClass: {classes[target]}\nIdx: {idx}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("./plots/lowest_npz_mem_images.png", dpi=300)
    plt.close()

    print("Images saved to ./plots/highest_npz_mem_images.png and ./plots/lowest_npz_mem_images.png")

if __name__ == "__main__":
    main()
