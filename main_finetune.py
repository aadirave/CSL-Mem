import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return index, data, target

    def __len__(self):
        return len(self.dataset)


def train_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    learning_rate = 0.1
    epochs_phase1 = 50
    epochs_phase2 = 50
    total_epochs = epochs_phase1 + epochs_phase2

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ]
    )

    base_trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    indexed_trainset = IndexedDataset(base_trainset)
    full_loader = torch.utils.data.DataLoader(
        indexed_trainset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    torch.manual_seed(42)
    indices = torch.randperm(len(base_trainset))
    part_a_indices = indices[:len(indices)//2]
    part_b_indices = indices[len(indices)//2:]
    
    trainset_a = torch.utils.data.Subset(indexed_trainset, part_a_indices)
    trainset_b = torch.utils.data.Subset(indexed_trainset, part_b_indices)

    trainloader_a = torch.utils.data.DataLoader(
        trainset_a, batch_size=batch_size, shuffle=True, num_workers=2
    )
    trainloader_b = torch.utils.data.DataLoader(
        trainset_b, batch_size=batch_size, shuffle=True, num_workers=2
    )

    num_train_samples = len(indexed_trainset)
    loss_tracker = torch.zeros(total_epochs, num_train_samples)
    grad_norm_tracker = torch.zeros(total_epochs, num_train_samples)

    model = resnet50(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    print(f"Starting finetune training on {device}...")
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        
        current_loader = trainloader_a if epoch < epochs_phase1 else trainloader_b
        
        for i, (_, inputs, targets) in enumerate(current_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{total_epochs} [{'Phase 1 (Partition A)' if epoch < epochs_phase1 else 'Phase 2 (Partition B)'}] Train Loss: {running_loss / len(current_loader):.4f}")

        # Tracking Pass for All Samples
        model.eval()
        with torch.enable_grad():
            for idx, inputs, targets in full_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs.requires_grad_(True)
                inputs.retain_grad()

                model.zero_grad()
                outputs = model(inputs)
                per_sample_loss = criterion(outputs, targets)
                loss = per_sample_loss.mean()
                loss.backward()

                sample_grad = inputs.grad * inputs.size(0)
                grad_norm = sample_grad.view(sample_grad.size(0), -1).norm(dim=1)

                loss_tracker[epoch, idx] = per_sample_loss.detach().cpu()
                grad_norm_tracker[epoch, idx] = grad_norm.detach().cpu()

    os.makedirs("./pretrained", exist_ok=True)
    torch.save(model.state_dict(), "./pretrained/resnet50_finetune.pt")
    os.makedirs("./logs", exist_ok=True)
    torch.save(
        {"loss_tracker": loss_tracker, "grad_norm_tracker": grad_norm_tracker},
        "./logs/tracking_data_finetune.pt",
    )
    print("Saved tracking_data_finetune.pt")

if __name__ == "__main__":
    train_finetune()
