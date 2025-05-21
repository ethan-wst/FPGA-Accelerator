# MobileNetV3 Inference Benchmark Script
# Goal: Evaluate inference performance of pretrained MobileNetV3 models on dummy input
#       This script avoids dataset training to isolate hardware speed performance

import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
import argparse
import time
import csv
import os

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Benchmark MobileNetV3 inference speed')
parser.add_argument('--model', type=str, default='mobilenet_v3_small', help='Model type (mobilenet_v3_small or mobilenet_v3_large)')
parser.add_argument('--input_size', type=int, default=256, help='Height/width of input tensor')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dummy input')
parser.add_argument('--iterations', type=int, default=100, help='Number of inference iterations')
args = parser.parse_args()

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# --- Load Model ---
if args.model == 'mobilenet_v3_large':
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
else:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

model.eval()    # Puts model into inference mode
model.to(device)    # Moves the model to selected device

# --- Generate Dummy Input ---
input_tensor = torch.randn(args.batch_size, 3, args.input_size, args.input_size).to(device)

# --- Warm-up ---
# Allows kernal initialization, CPU caching, leads to more consistent timing
for _ in range(10):
    _ = model(input_tensor)

# --- Benchmark ---
times = []
with torch.no_grad():
    for _ in range(args.iterations):
        start = time.time()
        _ = model(input_tensor)
        end = time.time()
        times.append((end - start) / args.batch_size)

avg_time = sum(times) / len(times) * 1000  # ms/image
print(f"Model: {args.model}")
print(f"Input Size: {args.input_size}x{args.input_size}, Batch Size: {args.batch_size}")
print(f"Average Inference Time: {avg_time:.2f} ms/image over {args.iterations} iterations")

# --- Data Export ---
csv_path = "benchmark_results.csv"
write_header = not os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Model", "Input Size", "Batch Size", "Avg Inference Time (ms)"])
    writer.writerow([args.model, args.input_size, args.batch_size, f"{avg_time:.2f}"])

