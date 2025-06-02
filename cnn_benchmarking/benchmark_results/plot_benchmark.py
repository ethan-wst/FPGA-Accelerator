import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
csv_path = os.path.join(os.path.dirname(__file__), 'onnx_local_benchmark_results.csv')

# Read CSV with header
with open(csv_path, 'r') as f:
    first_line = f.readline()
    if 'Model' not in first_line:
        columns = ["Device", "Model", "Format", "Input Size", "Avg Inference Time (ms)", 
                   "Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)"]
        df = pd.read_csv(csv_path, names=columns)
    else:
        df = pd.read_csv(csv_path)

# Clean up whitespace in column names
if df.columns[0].startswith('Device'):
    df.columns = [c.strip() for c in df.columns]

# Remove duplicates
if 'Device' in df.columns and 'Model' in df.columns:
    df = df.drop_duplicates(subset=["Device", "Model"], keep='last')
else:
    df = df.drop_duplicates()

# Convert columns to numeric
for col in ["Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)", "Avg Inference Time (ms)"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Split by device
cpu_df = df[df['Device'] == 'cpu']
cuda_df = df[df['Device'] == 'cuda']

# --- 1. Bubble map: Avg Inference Time vs Top-1 Accuracy (bubble size = model size) ---
def plot_bubble_map(device_df, device_name):
    plt.figure(figsize=(12, 8))
    x = device_df["Avg Inference Time (ms)"]
    y = device_df["Top-1 Accuracy (%)"]
    sizes = device_df["Total Size (MB)"] * 100  # scale for visibility
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(device_df['Model'].unique())))
    color_map = {name: colors[i] for i, name in enumerate(device_df['Model'].unique())}
    color_list = [color_map[m] for m in device_df['Model']]
    
    plt.scatter(x, y, s=sizes, c=color_list, alpha=0.7, edgecolors='w', linewidths=1)

    for i, row in device_df.iterrows():
        plt.text(row["Avg Inference Time (ms)"], row["Top-1 Accuracy (%)"], row["Model"], 
                 fontsize=9, ha='center', va='center')
    
    plt.xlabel("Avg Inference Time (ms)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title(f"CNN Models on {device_name.upper()}: Inference Time vs. Top-1 Accuracy (Bubble Size = Model Size MB)")

    # Set consistent y-axis range from 65% to 100%
    plt.ylim(65, 100)
    plt.xlim(0, 25)
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"benchmark_bubble_map_{device_name}.png")
    plt.close()

# --- Generate plots for each device ---
if not cpu_df.empty:
    plot_bubble_map(cpu_df, "cpu")
    plot_bubble_map(cuda_df, "cuda")