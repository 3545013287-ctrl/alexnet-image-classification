import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from alexnet_model import build_alexnet


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned AlexNet checkpoint on the test set.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_alexnet.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(num_classes):
    return build_alexnet(num_classes=num_classes, pretrained=False)


def build_test_loader(data_dir, batch_size, num_workers):
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_dataset, test_loader


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, device)
    class_to_idx = checkpoint["class_to_idx"]

    test_dataset, test_loader = build_test_loader(args.data_dir, args.batch_size, args.num_workers)
    if test_dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "Class order in dataset/test does not match checkpoint. "
            f"dataset={test_dataset.class_to_idx}, checkpoint={class_to_idx}"
        )

    model = build_model(len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(len(class_to_idx), dtype=torch.long)
    class_total = torch.zeros(len(class_to_idx), dtype=torch.long)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            for label, pred in zip(labels.cpu(), preds.cpu()):
                class_total[label] += 1
                class_correct[label] += int(label == pred)

    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Test folder: {(args.data_dir / 'test').resolve()}")
    print(f"Device: {device}")
    print(f"Classes: {test_dataset.classes}")
    print(f"Class mapping: {test_dataset.class_to_idx}")
    print(f"Test images: {total_samples}")
    print(f"Loss: {total_loss / total_samples:.4f}")
    print(f"Accuracy: {total_correct / total_samples:.4f} ({total_correct}/{total_samples})")
    print("Per-class accuracy:")
    for idx, class_name in enumerate(test_dataset.classes):
        total = class_total[idx].item()
        correct = class_correct[idx].item()
        acc = correct / total if total else 0.0
        print(f"  {class_name}: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
