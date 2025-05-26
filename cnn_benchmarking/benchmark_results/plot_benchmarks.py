# This script generates two plots from the benchmark results:
# 1. Bar chart for model efficiency (Top-1 and Top-5 accuracy per MB)
# 2. Bubble map: Avg Inference Time vs Top-1 Accuracy (bubble size = model size)

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
# Try both possible locations for benchmark_results.csv
csv_paths = [
    os.path.join(os.path.dirname(__file__), 'benchmark_results.csv'),
    os.path.join(os.path.dirname(__file__), '../benchmark_results/benchmark_results.csv')
]
csv_path = None
for path in csv_paths:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        csv_path = path
        break
if csv_path is None:
    raise RuntimeError("No benchmark results found in benchmark_results.csv. Run a benchmark script first.")

# Read CSV with header
with open(csv_path, 'r') as f:
    first_line = f.readline()
    if 'Model' not in first_line:
        columns = ["Device", "Model", "Input Size", "Avg Inference Time (ms)", "Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)"]
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

# --- 1. Bar chart for model efficiency ---
# Calculate efficiency
for col in ["Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['Top-1 per MB'] = df["Top-1 Accuracy (%)"] / df["Total Size (MB)"]
df['Top-5 per MB'] = df["Top-5 Accuracy (%)"] / df["Total Size (MB)"]

# Sort by Top-1 per MB for better visualization
df_sorted = df.sort_values('Top-1 per MB', ascending=False)

plt.figure(figsize=(14, 7))
bar_width = 0.35
x = np.arange(len(df_sorted))
plt.bar(x - bar_width/2, df_sorted['Top-1 per MB'], width=bar_width, label='Top-1 per MB', color='mediumseagreen')
plt.bar(x + bar_width/2, df_sorted['Top-5 per MB'], width=bar_width, label='Top-5 per MB', color='orange')
plt.xticks(x, df_sorted['Model'], rotation=20, ha='right')
plt.ylabel('Accuracy per MB')
plt.xlabel('Model')
plt.title('Model Efficiency: Accuracy per Model Size (MB)')
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_efficiency_bar.png')
plt.show()

# --- 2. Bubble map: Avg Inference Time vs Top-1 Accuracy (bubble size = model size) ---
plt.figure(figsize=(12, 8))
x = df["Avg Inference Time (ms)"].astype(float)
y = df["Top-1 Accuracy (%)"].astype(float)
sizes = df["Total Size (MB)"].astype(float) * 10  # scale for visibility
colors = plt.get_cmap('tab20', len(df['Model'].unique()))
color_map = {name: colors(i) for i, name in enumerate(df['Model'].unique())}
color_list = [color_map[m] for m in df['Model']]
plt.scatter(x, y, s=sizes, c=color_list, alpha=0.7, edgecolors='w', linewidths=1)
for i, row in df.iterrows():
    plt.text(row["Avg Inference Time (ms)"], row["Top-1 Accuracy (%)"], row["Model"], fontsize=9, ha='center', va='center')
plt.xlabel("Avg Inference Time (ms)")
plt.ylabel("Top-1 Accuracy (%)")
plt.title("CNN Models: Inference Time vs. Top-1 Accuracy (Bubble Size = Model Size MB)")
# Custom legend for model colors
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[m], markersize=10, label=m) for m in df['Model']]
plt.legend(handles=handles, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("benchmark_bubble_map.png")
plt.show()
