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


def train_phase1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    learning_rate = 0.1
    epochs = 100

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
    trainset = IndexedDataset(base_trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    num_train_samples = len(trainset)
    loss_tracker = torch.zeros(epochs, num_train_samples)
    grad_norm_tracker = torch.zeros(epochs, num_train_samples)

    # cifar100 images are 32x32
    model = resnet50(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (indices, inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad_(True)
            inputs.retain_grad()

            optimizer.zero_grad()
            outputs = model(inputs)
            per_sample_loss = criterion(outputs, targets)
            loss = per_sample_loss.mean()
            loss.backward()

            sample_grad = inputs.grad * inputs.size(0)
            grad_norm = sample_grad.view(sample_grad.size(0), -1).norm(dim=1)

            loss_tracker[epoch, indices] = per_sample_loss.detach().cpu()
            grad_norm_tracker[epoch, indices] = grad_norm.detach().cpu()

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(trainloader):.4f}"
        )

    os.makedirs("./pretrained", exist_ok=True)
    torch.save(model.state_dict(), "./pretrained/resnet50_base.pt")
    print("Phase 1 Complete. Base model saved to ./pretrained/resnet50_base.pt")

    os.makedirs("./logs", exist_ok=True)
    torch.save(
        {"loss_tracker": loss_tracker, "grad_norm_tracker": grad_norm_tracker},
        "./logs/tracking_data.pt",
    )
    print("Tracking data saved to ./logs/tracking_data.pt")


if __name__ == "__main__":
    train_phase1()
