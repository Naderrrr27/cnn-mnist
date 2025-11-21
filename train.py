import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model_def import CNN, MODEL_PATH, DEVICE

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST("./data", train=True, transform=transform, download=True)
test_ds = datasets.MNIST("./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 200 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch}, Average loss: {avg_loss:.4f}")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    acc = correct / total
    print(f"Test loss: {test_loss/len(test_loader):.4f}, Accuracy: {acc*100:.2f}%")
    return acc

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        evaluate()

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
