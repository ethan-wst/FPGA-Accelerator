# This script generates plots from benchmark results, separated by device (CPU/CUDA)
# 1. Bar charts for model efficiency (Top-1 and Top-5 accuracy per MB)
# 2. Bubble maps: Avg Inference Time vs Top-1 Accuracy (bubble size = model size)

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
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

# Convert columns to numeric
for col in ["Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)", "Avg Inference Time (ms)"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate efficiency metrics
df['Top-1 per MB'] = df["Top-1 Accuracy (%)"] / df["Total Size (MB)"]
df['Top-5 per MB'] = df["Top-5 Accuracy (%)"] / df["Total Size (MB)"]

# Split by device
cpu_df = df[df['Device'] == 'cpu']
cuda_df = df[df['Device'] == 'cuda']

# --- 1. Bar chart for model efficiency (separated by device) ---
def plot_efficiency_bar(device_df, device_name):
    # Sort by Top-1 per MB for better visualization
    df_sorted = device_df.sort_values('Top-1 per MB', ascending=False)
    
    plt.figure(figsize=(14, 7))
    bar_width = 0.35
    x = np.arange(len(df_sorted))
    plt.bar(x - bar_width/2, df_sorted['Top-1 per MB'], width=bar_width, label='Top-1 per MB', color='mediumseagreen')
    plt.bar(x + bar_width/2, df_sorted['Top-5 per MB'], width=bar_width, label='Top-5 per MB', color='orange')
    plt.xticks(x, df_sorted['Model'], rotation=20, ha='right')
    plt.ylabel('Accuracy per MB')
    plt.xlabel('Model')
    plt.title(f'Model Efficiency on {device_name.upper()}: Accuracy per Model Size (MB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'benchmark_efficiency_bar_{device_name}.png')
    plt.close()

# --- 2. Bubble map: Avg Inference Time vs Top-1 Accuracy (bubble size = model size) ---
def plot_bubble_map(device_df, device_name):
    plt.figure(figsize=(12, 8))
    x = device_df["Avg Inference Time (ms)"]
    y = device_df["Top-1 Accuracy (%)"]
    sizes = device_df["Total Size (MB)"] * 10  # scale for visibility
    
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
    
    # Custom legend for model colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[m], 
                         markersize=10, label=m) for m in device_df['Model']]
    plt.legend(handles=handles, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"benchmark_bubble_map_{device_name}.png")
    plt.close()

# Generate plots for each device
if not cpu_df.empty:
    plot_efficiency_bar(cpu_df, "cpu")
    plot_bubble_map(cpu_df, "cpu")
    
if not cuda_df.empty:
    plot_efficiency_bar(cuda_df, "cuda")
    plot_bubble_map(cuda_df, "cuda")

# --- 3. Combined comparison plots for CPU vs CUDA ---
# Only create if we have both CPU and CUDA data
if not cpu_df.empty and not cuda_df.empty:
    # Speed comparison for common models
    common_models = set(cpu_df['Model']).intersection(set(cuda_df['Model']))
    if common_models:
        compare_df = pd.DataFrame()
        
        for model in common_models:
            cpu_time = cpu_df[cpu_df['Model'] == model]['Avg Inference Time (ms)'].values[0]
            cuda_time = cuda_df[cuda_df['Model'] == model]['Avg Inference Time (ms)'].values[0]
            speedup = cpu_time / cuda_time if cuda_time > 0 else 0
            
            # Create a single-row DataFrame for this model
            model_df = pd.DataFrame({
                'Model': [model],
                'CPU Time (ms)': [cpu_time],
                'CUDA Time (ms)': [cuda_time],
                'Speedup': [speedup]
            })
            
            # Concatenate with the main DataFrame
            compare_df = pd.concat([compare_df, model_df], ignore_index=True)
        
        # Sort by speedup
        compare_df = compare_df.sort_values('Speedup', ascending=False)
        
        plt.figure(figsize=(14, 8))
        x = np.arange(len(compare_df))
        width = 0.35
        
        # Grouped bar chart showing CPU vs CUDA times
        plt.bar(x - width/2, compare_df['CPU Time (ms)'], width, label='CPU')
        plt.bar(x + width/2, compare_df['CUDA Time (ms)'], width, label='CUDA')
        
        plt.xlabel('Model')
        plt.ylabel('Inference Time (ms)')
        plt.title('CPU vs CUDA Inference Time Comparison')
        plt.xticks(x, compare_df['Model'], rotation=20, ha='right')
        plt.legend()
        
        # Add speedup as text above bars
        for i, (_, row) in enumerate(compare_df.iterrows()):
            plt.text(i, max(row['CPU Time (ms)'], row['CUDA Time (ms)']) + 3, 
                    f"{row['Speedup']:.2f}x", ha='center')
        
        plt.tight_layout()
        plt.savefig('cpu_vs_cuda_inference_time.png')
        plt.close()

print(f"Plots generated successfully in: {os.path.dirname(csv_path)}")
