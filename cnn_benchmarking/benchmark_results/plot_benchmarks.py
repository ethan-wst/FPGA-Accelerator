# This script generates a bar plot from the benchmark results of different CNN models.
import os
import pandas as pd
import matplotlib.pyplot as plt

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

# Try to infer columns if header is missing
with open(csv_path, 'r') as f:
    first_line = f.readline()
    if 'Model' not in first_line:
        # Add header for legacy/no-header CSVs
        columns = ["Model", "Input Size", "Avg Inference Time (ms)", "Top-1 Accuracy (%)", "Top-5 Accuracy (%)"]
        df = pd.read_csv(csv_path, names=columns)
    else:
        df = pd.read_csv(csv_path)

# Optional: remove duplicates
df = df.drop_duplicates()

# Plotting
plt.figure(figsize=(10, 6))

bar_width = 0.25
x = range(len(df["Model"]))

ax = plt.gca()

# Inference time (left y-axis)
bars1 = ax.bar([i - bar_width for i in x], df["Avg Inference Time (ms)"].astype(float), width=bar_width, label='Avg Inference Time (ms)', color='skyblue')
ax.set_ylabel('Avg Inference Time (ms)', color='skyblue')
ax.tick_params(axis='y', labelcolor='skyblue')

# Accuracy (right y-axis)
ax2 = ax.twinx()
bars2 = ax2.bar(x, df["Top-1 Accuracy (%)"].astype(float), width=bar_width, label='Top-1 Accuracy (%)', color='mediumseagreen')
if "Top-5 Accuracy (%)" in df.columns:
    bars3 = ax2.bar([i + bar_width for i in x], df["Top-5 Accuracy (%)"].astype(float), width=bar_width, label='Top-5 Accuracy (%)', color='orange')
ax2.set_ylabel('Accuracy (%)', color='mediumseagreen')
ax2.tick_params(axis='y', labelcolor='mediumseagreen')
ax2.set_ylim(0, 100)

plt.xticks(x, df["Model"])
ax.set_xlabel("Model")
plt.title("CNN Benchmark: Inference Time and Accuracy")

# Combine legends from both axes
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

ax.grid(True, linestyle="--", alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig("benchmark_plot.png")
plt.show()
