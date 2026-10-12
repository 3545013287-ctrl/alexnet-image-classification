import argparse
import json
import random
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from alexnet_model import build_alexnet


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AlexNet on a small ImageFolder dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"), help="Dataset root with train/ and test/.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-conv-layers",
        type=int,
        default=5,
        choices=range(0, 6),
        metavar="{0,1,2,3,4,5}",
        help="Freeze the first N AlexNet convolution layers. 5 freezes all conv layers, 0 fine-tunes all layers.",
    )
    parser.add_argument(
        "--unfreeze-features",
        action="store_true",
        help="Compatibility shortcut for --freeze-conv-layers 0.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--augment",
        nargs="+",
        default=["random_crop", "hflip"],
        choices=["none", "random_crop", "hflip", "color_jitter", "rotation"],
        help="Training augmentations. Use 'none' for Resize + CenterCrop only.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_train_transform(augmentations):
    if "none" in augmentations:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    transform_steps = [transforms.Resize(256)]
    if "random_crop" in augmentations:
        transform_steps.append(transforms.RandomResizedCrop(224))
    else:
        transform_steps.append(transforms.CenterCrop(224))

    if "hflip" in augmentations:
        transform_steps.append(transforms.RandomHorizontalFlip())
    if "color_jitter" in augmentations:
        transform_steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    if "rotation" in augmentations:
        transform_steps.append(transforms.RandomRotation(15))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(transform_steps)


def build_dataloaders(data_dir, batch_size, num_workers, augmentations):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected dataset/train and dataset/test folders.")

    train_transform = build_train_transform(augmentations)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, test_dataset, train_loader, test_loader


def freeze_first_conv_layers(model, num_frozen_conv_layers):
    conv_layer_indices = [0, 3, 6, 8, 10]
    frozen_layer_indices = set(conv_layer_indices[:num_frozen_conv_layers])

    for layer_index in frozen_layer_indices:
        for param in model.features[layer_index].parameters():
            param.requires_grad = False

    frozen_names = [f"features[{idx}]" for idx in conv_layer_indices[:num_frozen_conv_layers]]
    trainable_names = [f"features[{idx}]" for idx in conv_layer_indices[num_frozen_conv_layers:]]
    return frozen_names, trainable_names


def build_model(num_classes, frozen_conv_layers):
    model = build_alexnet(
        num_classes=num_classes,
        pretrained=True,
        freeze_features=False,
    )
    frozen_names, trainable_names = freeze_first_conv_layers(model, frozen_conv_layers)
    return model, frozen_names, trainable_names


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(path, model, class_to_idx, epoch, best_acc, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "epoch": epoch,
            "best_acc": best_acc,
            "args": saved_args,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, args.augment
    )

    frozen_conv_layers = 0 if args.unfreeze_features else args.freeze_conv_layers
    model, frozen_names, trainable_names = build_model(len(train_dataset.classes), frozen_conv_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(train_dataset.class_to_idx, f, indent=2, ensure_ascii=False)

    print(f"Device: {device}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Train augmentations: {args.augment}")
    print(f"Frozen conv layers: {frozen_conv_layers}/5 {frozen_names}")
    print(f"Trainable conv layers: {trainable_names}")
    print(f"Train images: {len(train_dataset)} | Test images: {len(test_dataset)}")

    best_acc = 0.0
    best_path = args.output_dir / "best_alexnet.pth"
    last_path = args.output_dir / "last_alexnet.pth"

    for epoch in range(1, args.epochs + 1):
        start = time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_one_epoch(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(best_path, model, train_dataset.class_to_idx, epoch, best_acc, args)

        save_checkpoint(last_path, model, train_dataset.class_to_idx, epoch, best_acc, args)
        elapsed = time() - start
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"best_acc={best_acc:.4f} time={elapsed:.1f}s"
        )

    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
