import torch
import os

os.makedirs("./index", exist_ok=True)

num_samples = 50000

torch.manual_seed(42)
indices = torch.randperm(num_samples)

torch.save(indices, "./index/data_index_cifar10.pt")
torch.save(indices, "./index/data_index_cifar10_duplicate.pt")

torch.save(indices, "./index/data_index_cifar100.pt")

print("Successfully generated index files in ./index/")