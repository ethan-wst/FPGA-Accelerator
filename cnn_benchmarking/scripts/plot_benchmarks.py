# This script generates a bar plot from the benchmark results of different CNN models.
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("benchmark_results.csv")

# Optional: remove duplicates
df = df.drop_duplicates()

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(df["Model"], df["Avg Inference Time (ms)"].astype(float))

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + .2, f"{height:.1f}", ha='center')

plt.title("CNN Inference Benchmark")
plt.xlabel("Model")
plt.ylabel("Average Inference Time (ms/image)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("benchmark_plot.png")
plt.show()
