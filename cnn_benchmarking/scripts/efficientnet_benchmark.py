# efficientnet_benchmark.py
# Benchmark pretrained EfficientNet (Lite) using timm on dummy input
# python efficientnet_benchmark.py --model efficientnet_lite{0-4} --input_size 256 --batch_size 1 --iterations 100


import torch
import timm
import argparse
import time
import csv
import os


# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Benchmark EfficientNet-Lite inference speed')
parser.add_argument('--model', type=str, default='efficientnet_lite0', help='Efficientnet variant: efficientnet_lite0 - efficientnet_lite4')
parser.add_argument('--input_size', type=int, default=256, help='Input image size (height and width)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dummy input')
parser.add_argument('--iterations', type=int, default=100, help='Number of inference iterations')
args = parser.parse_args()

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# --- Load Model ---
model = timm.create_model(args.model, pretrained=True)
model.eval()
model.to(device)

# --- Generate Dummy Input ---
input_tensor = torch.randn(args.batch_size, 3, args.input_size, args.input_size).to(device)

# --- Warm-up ---
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

