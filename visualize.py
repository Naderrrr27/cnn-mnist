import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.model_def import CNN, MODEL_PATH, DEVICE

if not os.path.exists(MODEL_PATH):
    print(f"Model file '{MODEL_PATH}' not found. Please run train.py first.")
    sys.exit()

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_ds = datasets.MNIST("./data", train=False, transform=transform, download=False)

def visualize():
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    weights = model.conv1.weight.detach().cpu().numpy()
    n_filters = weights.shape[0]
    plt.figure(figsize=(8,2))
    for i in range(n_filters):
        filt = weights[i,0]
        plt.subplot(1, n_filters, i+1)
        plt.imshow(filt, cmap='gray')
        plt.axis('off')
        plt.title(f"F{i}")
    plt.suptitle("conv1 filters")
    plt.show()

    idx = torch.randint(0, len(test_ds), (1,)).item()
    img, label = test_ds[idx]

    with torch.no_grad():
        logits, feat1, feat2 = model(img.unsqueeze(0).to(DEVICE), return_feats=True)

    print(f"Label: {label}, Predicted: {logits.argmax().item()}")

    plt.figure()
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Real Label: {label}")
    plt.axis('off')
    plt.show()

    feat1 = feat1.squeeze(0).numpy()
    feat2 = feat2.squeeze(0).numpy()

    def plot_maps(maps, title, max_plots=8):
        n = min(maps.shape[0], max_plots)
        plt.figure(figsize=(12, 2))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(maps[i], cmap='viridis')
            plt.axis('off')
            plt.title(f"M{i}")
        plt.suptitle(title)
        plt.show()

    plot_maps(feat1, "Feature maps after conv1 (ReLU)", max_plots=8)
    plot_maps(feat2, "Feature maps after conv2 (ReLU)", max_plots=8)


if __name__ == "__main__":
    visualize()
