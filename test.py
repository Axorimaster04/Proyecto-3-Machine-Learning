#!/usr/bin/env python3
"""
PyTorch Butterfly Classifier
- Loads a Kaggle butterflies dataset (folder-per-class) and filters only colored images
- Splits into 80/10/10 (train/val/test) with stratification
- Trains a configurable MLP (hidden layers, activation: relu|tanh|sigmoid) for multi-class classification
- Optional: simple CNN for comparison (set --model cnn)
- Records metrics: accuracy, precision, recall, F1 (macro & per-class), confusion matrix
- Visualizes loss and accuracy/F1 curves (train/val)

Usage example:
    python train_butterflies.py \
        --data_dir /path/to/butterflies \
        --out_dir ./runs/butterflies_mlp \
        --model mlp \
        --img_size 128 \
        --hidden_sizes 512 256 \
        --activation relu \
        --optimizer adam \
        --epochs 25 --batch_size 64 --lr 3e-4

CNN example:
    python train_butterflies.py --model cnn --img_size 128 --epochs 20

Folder structure expected by torchvision.datasets.ImageFolder:
    data_dir/
       class_1/ img1.jpg, img2.jpg, ...
       class_2/ ...
       ...

Outputs (in out_dir):
    best_model.pt
    history.json
    curves_loss.png
    curves_acc_f1.png
    confusion_matrix.png
    classification_report.txt
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

import kagglehub


def download_dataset():
    print("Downloading Butterfly dataset from Kaggle...")
    path = kagglehub.dataset_download("veeralakrishna/butterfly-dataset")
    print("Path to dataset files:", path)
    return path

# ---------------------------- Utils ---------------------------- #

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------- Dataset & Splits -------------------- #

def is_colored_image(path: str) -> bool:
    """Return True if image appears to be colored (3 channels)."""
    try:
        with Image.open(path) as im:
            return im.mode in ("RGB", "RGBA")
    except Exception:
        return False


def filter_colored_samples(dataset: ImageFolder) -> List[Tuple[str, int]]:
    colored = []
    for p, y in dataset.samples:
        if is_colored_image(p):
            colored.append((p, y))
    return colored


def stratified_indices(labels: List[int], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    classes = np.unique(labels)

    train_idx, val_idx, test_idx = [], [], []

    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    # Shuffle each split to mix classes
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


# -------------------- Models -------------------- #

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_sizes: List[int], activation: str, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        Act = ACTIVATIONS[activation]
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), Act(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# -------------------- Training / Eval -------------------- #

@dataclass
class History:
    train_loss: List[float]
    val_loss: List[float]
    val_acc: List[float]
    val_f1_macro: List[float]


def make_loaders(data_dir: str, img_size: int, batch_size: int, seed: int = 42):
    # Base dataset to read files & labels
    base = ImageFolder(data_dir)
    # Filter only colored images
    colored_samples = filter_colored_samples(base)
    if len(colored_samples) == 0:
        raise RuntimeError("No colored images found. Check your dataset path or filtering.")
    base.samples = colored_samples
    base.imgs = colored_samples

    # Extract labels for stratified split
    labels = [y for _, y in base.samples]
    train_idx, val_idx, test_idx = stratified_indices(labels, seed=seed)

    # Transforms
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ToTensor(),
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    # Clone datasets per split so we can apply different transforms
    train_ds = ImageFolder(data_dir, transform=train_tf)
    val_ds = ImageFolder(data_dir, transform=eval_tf)
    test_ds = ImageFolder(data_dir, transform=eval_tf)

    # Reuse the filtered colored samples for each split
    for ds in (train_ds, val_ds, test_ds):
        ds.samples = colored_samples
    ds.imgs = colored_samples

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=2,
                            pin_memory=True)
    test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=2,
                             pin_memory=True)

    return train_loader, val_loader, test_loader, base.classes


def build_model(model_name: str, img_size: int, num_classes: int, hidden_sizes: List[int], activation: str, dropout: float):
    if model_name == "mlp":
        input_dim = 3 * img_size * img_size
        model = MLP(input_dim=input_dim, num_classes=num_classes, hidden_sizes=hidden_sizes, activation=activation, dropout=dropout)
    elif model_name == "cnn":
        model = SimpleCNN(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError("model must be 'mlp' or 'cnn'")
    return model


def get_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")


def train_one_epoch(model, loader, criterion, optimizer, device, model_name: str):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if model_name == "mlp":
            x = torch.flatten(x, 1)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, model_name: str):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if model_name == "mlp":
                x = torch.flatten(x, 1)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, prec, rec, f1, y_true, y_pred


def plot_curves(history: History, out_dir: str):
    # Loss curves
    plt.figure()
    plt.plot(history.train_loss, label="train_loss")
    plt.plot(history.val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss (train vs val)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_loss.png"))
    plt.close()

    # Accuracy & F1 curves
    plt.figure()
    plt.plot(history.val_acc, label="val_acc")
    plt.plot(history.val_f1_macro, label="val_f1_macro")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Validation Accuracy & F1 (macro)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_acc_f1.png"))
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names: List[str], out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# -------------------- Main -------------------- #

def main():
    parser = argparse.ArgumentParser(description="Butterfly classifier (MLP/CNN) in PyTorch")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset (optional: will auto-download if not provided)')
    parser.add_argument('--out_dir', type=str, default='./runs/butterflies', help='Output directory')

    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'])
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--hidden_sizes', type=int, nargs='*', default=[512, 256], help='MLP hidden layer sizes')
    parser.add_argument('--activation', type=str, default='relu', choices=list(ACTIVATIONS.keys()))
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stopping', type=int, default=7, help='Patience (epochs) without val loss improvement before stop. 0 disables.')

    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_out_dir(args.out_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.data_dir is None:
        data_dir = download_dataset()
    else:
        data_dir = args.data_dir


    data_dir = "C:\\Users\\RZAMBRAN\\.cache\\kagglehub\\datasets\\veeralakrishna\\butterfly-dataset\\versions\\1\\leedsbutterfly\\organized"

    train_loader, val_loader, test_loader, class_names = make_loaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )


    # Model
    model = build_model(
        model_name=args.model,
        img_size=args.img_size,
        num_classes=len(class_names),
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    # Optimizer & Loss
    optimizer = get_optimizer(args.optimizer, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = History(train_loss=[], val_loss=[], val_acc=[], val_f1_macro=[])
    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1
    patience = args.early_stopping

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.model)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device, args.model)

        history.train_loss.append(tr_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.val_f1_macro.append(val_f1)

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        # Checkpoint best
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            torch.save(best_state, os.path.join(args.out_dir, 'best_model.pt'))
        else:
            # Early stopping
            if patience and (epoch - best_epoch) >= patience:
                print(f"Early stopping at epoch {epoch} (best val at {best_epoch})")
                break

    # Save history
    hist_path = os.path.join(args.out_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump({
            'train_loss': history.train_loss,
            'val_loss': history.val_loss,
            'val_acc': history.val_acc,
            'val_f1_macro': history.val_f1_macro,
        }, f, indent=2)

    plot_curves(history, args.out_dir)

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device, args.model)

    print(f"\nTest: loss={test_loss:.4f} acc={test_acc:.4f} prec_macro={test_prec:.4f} rec_macro={test_rec:.4f} f1_macro={test_f1:.4f}")

    # Save classification report & confusion matrix
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(args.out_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print("\nClassification Report:\n")
    print(report)

    save_confusion_matrix(y_true, y_pred, class_names, os.path.join(args.out_dir, 'confusion_matrix.png'))


if __name__ == '__main__':
    main()
