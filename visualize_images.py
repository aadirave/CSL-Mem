import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np

def imshow(img, title, filepath):
    # Unnormalize
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.savefig(filepath, dpi=300)
    plt.close()

def main():
    device = torch.device('cpu')
    tracking_file = "./logs/tracking_data.pt"
    
    if not os.path.exists(tracking_file):
        print(f"Error: {tracking_file} not found.")
        return

    print("Loading datasets...")
    # Load same test transforms/data without augmentation to see base images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # We must use training set because indices match the tracking data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes = trainset.classes

    print(f"Loading {tracking_file}...")
    data = torch.load(tracking_file, map_location=device)
    grad_norm_tracker = data['grad_norm_tracker'] 
    
    print("Calculating CSG scores...")
    csg_scores = (grad_norm_tracker ** 2).sum(dim=0)
    
    k = 5
    sorted_csg, sorted_indices = torch.sort(csg_scores)
    lowest_csg_indices = sorted_indices[:k]
    highest_csg_indices = sorted_indices[-k:]
    
    os.makedirs("./plots", exist_ok=True)
    
    print("\nVisualizing Lowest CSG Images...")
    fig_low = plt.figure(figsize=(15, 3))
    for i, idx in enumerate(lowest_csg_indices):
        img, target = trainset[idx.item()]
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        ax = fig_low.add_subplot(1, k, i + 1)
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(f"CSG: {csg_scores[idx]:.1f}\nClass: {classes[target]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./plots/lowest_csg_images.png', dpi=300)
    plt.close()

    print("Visualizing Highest CSG Images...")
    fig_high = plt.figure(figsize=(15, 3))
    for i, idx in enumerate(highest_csg_indices):
        img, target = trainset[idx.item()]
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        ax = fig_high.add_subplot(1, k, i + 1)
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(f"CSG: {csg_scores[idx]:.1f}\nClass: {classes[target]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./plots/highest_csg_images.png', dpi=300)
    plt.close()
    
    print("Images saved to ./plots/")

if __name__ == "__main__":
    main()
