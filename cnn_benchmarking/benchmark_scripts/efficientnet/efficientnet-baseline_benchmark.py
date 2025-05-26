# efficientnet-baseline_benchmark.py
# Benchmark pretrained EfficientNet-B0 to B7 on random images from ImageNet_SubSet to measure inference speed and top-1/top-5 accuracy

import torch
import torchvision.models as models
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights
)
import argparse
import time
import csv
import os
import random
from PIL import Image
from torchvision import transforms
import scipy.io as sio
import json

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Benchmark EfficientNet-B0 to B7 inference speed and accuracy')
parser.add_argument('--model', type=str, default='efficientnet_b0', help='Model name: efficientnet_b0 ... efficientnet_b7')
parser.add_argument('--imagenet_dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../imagenet/imagenet_subset')), help='Path to ImageNet_SubSet directory')
parser.add_argument('--num_images', type=int, default=500, help='Number of random images to use for benchmarking')
parser.add_argument('--meta', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../imagenet/ILSVRC2012_devkit_t12/data/meta.mat')), help='Path to meta.mat for WNID mapping')
args = parser.parse_args()

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# --- Load Model ---
weights_map = {
    'efficientnet_b0': EfficientNet_B0_Weights.DEFAULT,
    'efficientnet_b1': EfficientNet_B1_Weights.DEFAULT,
    'efficientnet_b2': EfficientNet_B2_Weights.DEFAULT,
    'efficientnet_b3': EfficientNet_B3_Weights.DEFAULT,
    'efficientnet_b4': EfficientNet_B4_Weights.DEFAULT,
    'efficientnet_b5': EfficientNet_B5_Weights.DEFAULT,
    'efficientnet_b6': EfficientNet_B6_Weights.DEFAULT,
    'efficientnet_b7': EfficientNet_B7_Weights.DEFAULT,
}
if args.model not in weights_map:
    raise ValueError(f"Unknown model: {args.model}")
weights = weights_map[args.model]
model = getattr(models, args.model)(weights=weights)
model.eval()
model.to(device)

# --- Load WNID to class index mapping ---
meta = sio.loadmat(args.meta)
wnids = [str(x[0]) for x in meta['synsets']['WNID'][0]]
wnid_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

# --- Build WNID to model class index mapping (for accuracy) ---
with open(os.path.join(os.path.dirname(__file__), '../../imagenet/imagenet_class_index.json'), 'r') as f:
    class_idx = json.load(f)
wnid_to_model_idx = {v[0]: int(k) for k, v in class_idx.items()}

# --- Gather all image paths and their WNIDs ---
image_paths = []
gt_wnids = []
for wnid in os.listdir(args.imagenet_dir):
    wnid_dir = os.path.join(args.imagenet_dir, wnid)
    if not os.path.isdir(wnid_dir):
        continue
    for fname in os.listdir(wnid_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(wnid_dir, fname))
            gt_wnids.append(wnid)

# --- Randomly sample images ---
if args.num_images < len(image_paths):
    sampled = random.sample(list(zip(image_paths, gt_wnids)), args.num_images)
else:
    sampled = list(zip(image_paths, gt_wnids))

# --- Determine model's preferred input size ---
if hasattr(model, 'default_cfg') and 'input_size' in model.default_cfg:
    input_height, input_width = model.default_cfg['input_size'][-2:]
else:
    input_height = input_width = 224  # Fallback

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((input_height, input_width)),
    transforms.CenterCrop((input_height, input_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Benchmark and Accuracy ---
times = []
correct = 0
top5_correct = 0
total = 0
with torch.no_grad():
    for img_path, wnid in sampled:
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            start = time.time()
            output = model(input_tensor)
            end = time.time()
            times.append((end - start))
            pred = output.argmax(dim=1).item()
            gt_idx = wnid_to_model_idx.get(wnid, -1)
            if pred == gt_idx:
                correct += 1
            # Top-5 accuracy
            top5 = output.topk(5, dim=1).indices.squeeze(0).tolist()
            if gt_idx in top5:
                top5_correct += 1
            total += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

avg_time = sum(times) / len(times) * 1000 if times else 0  # ms/image
accuracy = correct / total * 100 if total > 0 else 0
top5_accuracy = top5_correct / total * 100 if total > 0 else 0

total_params = sum(param.numel() for param in model.parameters())
total_size_bytes = total_params * 4  # float32
total_size_mb = total_size_bytes / 1024 / 1024

print(f"Model: {args.model}")
print(f"Input Size: {input_height}x{input_width}")
print(f"Average Inference Time: {avg_time:.2f} ms/image over {total} images")
print(f"Top-1 Accuracy: {accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
print(f"Total size in MB: {total_size_mb:.2f}")
# --- Data Export ---
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../benchmark_results/benchmark_results.csv'))
write_header = not os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Device", "Model", "Input Size", "Avg Inference Time (ms)", "Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)"])
    writer.writerow([device, args.model, f"{input_height}x{input_width}", f"{avg_time:.2f}", f"{accuracy:.2f}", f"{top5_accuracy:.2f}", f"{total_size_mb:.2f}"])
