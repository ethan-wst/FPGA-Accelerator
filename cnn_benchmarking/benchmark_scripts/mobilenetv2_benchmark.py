# mobilenetv2_benchmark.py
# Benchmark pretrained MobileNetV2 on random images from ImageNet_SubSet to measure inference speed and top-1 accuracy

import torch
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
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
parser = argparse.ArgumentParser(description='Benchmark MobileNetV2 inference speed and accuracy')
parser.add_argument('--model', type=str, default='mobilenet_v2', help='Model name for logging (default: mobilenet_v2)')
parser.add_argument('--input_size', type=int, default=256, help='Height/width of input tensor')
parser.add_argument('--imagenet_dir', type=str, default='../imagenet/imagenet_subset', help='Path to ImageNet_SubSet directory')
parser.add_argument('--num_images', type=int, default=100, help='Number of random images to use for benchmarking')
parser.add_argument('--meta', type=str, default='../imagenet/ILSVRC2012_devkit_t12/data/meta.mat', help='Path to meta.mat for WNID mapping')
args = parser.parse_args()

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# --- Load Model ---
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()
model.to(device)

# --- Load WNID to class index mapping ---
meta = sio.loadmat(args.meta)
wnids = [str(x[0]) for x in meta['synsets']['WNID'][0]]
wnid_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

# --- Build WNID to model class index mapping (for accuracy) ---
# Use official ImageNet class index mapping for torchvision models
with open(os.path.join(os.path.dirname(__file__), '../imagenet/imagenet_class_index.json'), 'r') as f:
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

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(args.input_size),
    transforms.CenterCrop(args.input_size),
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

print(f"Model: MobileNetV2")
print(f"Input Size: {args.input_size}x{args.input_size}")
print(f"Average Inference Time: {avg_time:.2f} ms/image over {total} images")
print(f"Top-1 Accuracy: {accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

# --- Data Export ---
csv_path = "../benchmark_results/benchmark_results.csv"
write_header = not os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Model", "Input Size", "Avg Inference Time (ms)", "Top-1 Accuracy (%)", "Top-5 Accuracy (%)"])
    writer.writerow([args.model, args.input_size, f"{avg_time:.2f}", f"{accuracy:.2f}", f"{top5_accuracy:.2f}"])