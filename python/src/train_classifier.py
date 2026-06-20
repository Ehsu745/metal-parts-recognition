"""
CNN 分類訓練腳本（對照組） — 金屬零件辨識

對應 MATLAB 階段的 googlenet_classification.m，此處改用 torchvision
預訓練模型重新實作分類路線，作為與 YOLOv8 物件偵測方法的比較對象。

注意：分類模型僅能判斷「整張圖是什麼類別」，無法像 YOLOv8 一樣標示
零件位置與計數多個個體。此腳本的目的是方法比較（速度、準確率、適用
情境），不是取代 YOLOv8 主線。

資料集結構需求（torchvision ImageFolder 格式）：
    data/cls_dataset/
        train/
            bolt/
            nut/
            screw/
            ...
        val/
            bolt/
            nut/
            screw/
            ...

執行：
    python train_classifier.py
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path(__file__).parent.parent / "data" / "cls_dataset"


def build_dataloaders(data_dir: Path, batch_size: int = 32, img_size: int = 224):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_set = datasets.ImageFolder(data_dir / "val", transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_set.classes


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    # 以 ResNet18 預訓練權重做遷移學習，作為起始點
    # （比起 GoogLeNet，torchvision 對 ResNet 的維護與文件更完整，
    #  之後若要換回 GoogLeNet 系列，torchvision 同樣提供 googlenet()）
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(epochs: int = 30, lr: float = 1e-4, batch_size: int = 32):
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"找不到資料集: {DATA_DIR}\n"
            "請先完成資料蒐集，並依 ImageFolder 格式整理至 python/data/cls_dataset/"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = build_dataloaders(DATA_DIR, batch_size=batch_size)

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {running_loss / len(train_loader):.4f} - val_acc: {val_acc:.4f}"
        )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金屬零件分類訓練（對照組）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
