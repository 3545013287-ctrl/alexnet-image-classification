import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from alexnet_model import build_alexnet


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one image with a fine-tuned AlexNet checkpoint.")
    parser.add_argument("image", type=Path, help="Path to an image file.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_alexnet.pth"))
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--visualize", action="store_true", help="Save a prediction visualization image.")
    parser.add_argument("--vis-output", type=Path, default=Path("outputs/predict_visualization.png"))
    parser.add_argument("--feature-channels", type=int, default=16)
    return parser.parse_args()


def build_model(num_classes):
    return build_alexnet(num_classes=num_classes, pretrained=False)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def preprocess_image(path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image = Image.open(path).convert("RGB")
    return image, transform(image).unsqueeze(0)


def denormalize_image(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def register_feature_hooks(model, feature_maps):
    layer_names = {
        0: "conv1",
        3: "conv2",
        6: "conv3",
        8: "conv4",
        10: "conv5",
    }
    hooks = []

    for layer_index, layer_name in layer_names.items():
        layer = model.features[layer_index]

        def hook(_, __, output, name=layer_name):
            feature_maps[name] = output.detach().cpu()

        hooks.append(layer.register_forward_hook(hook))

    return hooks


def make_feature_grid(feature_tensor, max_channels):
    feature_tensor = feature_tensor[0]
    channel_count = min(max_channels, feature_tensor.shape[0])
    channels = feature_tensor[:channel_count]
    channels = channels - channels.amin(dim=(1, 2), keepdim=True)
    channels = channels / (channels.amax(dim=(1, 2), keepdim=True) + 1e-6)

    cols = 4
    rows = (channel_count + cols - 1) // cols
    height, width = channels.shape[1:]
    grid = torch.ones(rows * height, cols * width)
    for idx, channel in enumerate(channels):
        row = idx // cols
        col = idx % cols
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = channel
    return grid.numpy()


def save_visualization(path, original_image, input_tensor, probabilities, idx_to_class, topk, feature_maps, max_channels):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    k = min(topk, len(idx_to_class))
    scores, indices = probabilities.topk(k)
    labels = [idx_to_class[index] for index in indices.tolist()]
    values = scores.tolist()

    layer_count = len(feature_maps)
    fig = plt.figure(figsize=(14, 4 + layer_count * 2.2))
    gs = fig.add_gridspec(layer_count + 1, 3)

    ax_original = fig.add_subplot(gs[0, 0])
    ax_original.imshow(original_image)
    ax_original.set_title("Original image")
    ax_original.axis("off")

    ax_input = fig.add_subplot(gs[0, 1])
    ax_input.imshow(denormalize_image(input_tensor[0]))
    ax_input.set_title("AlexNet input: 224x224")
    ax_input.axis("off")

    ax_probs = fig.add_subplot(gs[0, 2])
    ax_probs.barh(labels[::-1], values[::-1], color="#2b6cb0")
    ax_probs.set_xlim(0, 1)
    ax_probs.set_title("Top probabilities")
    ax_probs.set_xlabel("probability")

    for row, (layer_name, feature_tensor) in enumerate(feature_maps.items(), start=1):
        ax = fig.add_subplot(gs[row, :])
        ax.imshow(make_feature_grid(feature_tensor, max_channels), cmap="viridis")
        ax.set_title(f"{layer_name} feature maps")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    model = build_model(len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    feature_maps = {}
    hooks = register_feature_hooks(model, feature_maps) if args.visualize else []

    original_image, image = preprocess_image(args.image)
    image = image.to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)[0]

    for hook in hooks:
        hook.remove()

    k = min(args.topk, len(idx_to_class))
    scores, indices = probabilities.topk(k)
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {idx_to_class[index]}: {score:.4f}")

    if args.visualize:
        save_visualization(
            args.vis_output,
            original_image,
            image.cpu(),
            probabilities.cpu(),
            idx_to_class,
            args.topk,
            feature_maps,
            args.feature_channels,
        )
        print(f"Visualization saved to: {args.vis_output}")


if __name__ == "__main__":
    main()
