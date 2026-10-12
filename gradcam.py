import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from alexnet_model import build_alexnet


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM localization heatmap for AlexNet.")
    parser.add_argument("image", type=Path, help="Path to one image.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_alexnet.pth"))
    parser.add_argument("--class-name", type=str, default=None, help="Target class for Grad-CAM. Defaults to predicted class.")
    parser.add_argument("--output", type=Path, default=Path("outputs/gradcam.png"))
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity on the original image.")
    parser.add_argument("--box-threshold", type=float, default=0.55, help="Heatmap threshold used to draw a weak localization box.")
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def preprocess_image(path):
    display_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    tensor_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    original = Image.open(path).convert("RGB")
    model_input_image = display_transform(original)
    return original, model_input_image, tensor_transform(original).unsqueeze(0)


def build_model_from_checkpoint(checkpoint, device):
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    model = build_alexnet(num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    replace_inplace_relu(model)
    model.eval()
    return model, class_to_idx, idx_to_class


def replace_inplace_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_inplace_relu(child)


def compute_gradcam(model, image_tensor, target_index):
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = model.features[10]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(image_tensor)
    score = logits[:, target_index].sum()
    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    feature_map = activations["value"][0]
    gradient = gradients["value"][0]
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    cam = (weights * feature_map).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam.detach().cpu(), logits.detach()


def find_heatmap_box(cam_np, threshold):
    import numpy as np

    mask = cam_np >= threshold
    if not mask.any():
        max_y, max_x = np.unravel_index(cam_np.argmax(), cam_np.shape)
        return max_x, max_y, max_x + 1, max_y + 1

    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def save_gradcam(output_path, original_image, model_input_image, cam, predicted_class, probability, alpha, box_threshold):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_np = np.asarray(original_image).astype("float32") / 255.0
    input_np = np.asarray(model_input_image).astype("float32") / 255.0
    cam_image = Image.fromarray((cam.numpy() * 255).astype("uint8")).resize((224, 224), Image.BICUBIC)
    cam_np = np.asarray(cam_image).astype("float32") / 255.0

    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam_np)[..., :3]
    overlay = (1 - alpha) * input_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    box = find_heatmap_box(cam_np, box_threshold)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(input_np)
    axes[1].set_title("AlexNet input")
    axes[1].axis("off")

    axes[2].imshow(cam_np, cmap="jet")
    axes[2].set_title("Grad-CAM heatmap")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title(f"{predicted_class}: {probability:.4f}")
    x1, y1, x2, y2 = box
    axes[3].add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=2.5,
        )
    )
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return box


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, device)
    model, class_to_idx, idx_to_class = build_model_from_checkpoint(checkpoint, device)
    original_image, model_input_image, image_tensor = preprocess_image(args.image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    predicted_index = probabilities.argmax().item()

    if args.class_name is None:
        target_index = predicted_index
    else:
        if args.class_name not in class_to_idx:
            raise ValueError(f"Unknown class '{args.class_name}'. Available classes: {list(class_to_idx)}")
        target_index = class_to_idx[args.class_name]

    cam, logits = compute_gradcam(model, image_tensor, target_index)
    probabilities = torch.softmax(logits, dim=1)[0].cpu()
    target_class = idx_to_class[target_index]
    target_probability = probabilities[target_index].item()

    print(f"Image: {args.image}")
    print(f"Predicted: {idx_to_class[predicted_index]} ({probabilities[predicted_index].item():.4f})")
    print(f"Grad-CAM target: {target_class} ({target_probability:.4f})")

    box = save_gradcam(
        args.output,
        original_image,
        model_input_image,
        cam,
        target_class,
        target_probability,
        args.alpha,
        args.box_threshold,
    )
    print(f"Weak localization box on AlexNet input: x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}")
    print(f"Grad-CAM saved to: {args.output}")


if __name__ == "__main__":
    main()
