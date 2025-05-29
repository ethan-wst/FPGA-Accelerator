# efficientnet_benchmark.py
# Run all EfficientNet benchmarking scripts in the efficientnet subdirectory

import subprocess
import sys
import os

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), 'efficientnet')

# List of scripts and their model arguments
scripts_and_models = [
    # Baseline EfficientNet-B0 to B7
    # ('efficientnet-baseline_benchmark.py', [
    #     'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'
    # ]),
    # EfficientNet-Lite
    ('efficientnet-lite_benchmark.py', [
        'efficientnet_lite0'
    ]),
    # EfficientNet-Edge
    # ('efficientnet-edge_benchmark.py', [
    #     'efficientnet_es', 'efficientnet_em'
    # ]),
]

for script, models in scripts_and_models:
    script_path = os.path.join(SCRIPT_DIR, script)
    for model in models:
        print(f"\n===== Running {script} for model {model} =====")
        cmd = [sys.executable, script_path, '--model', model]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error running {script} for model {model}")
            break
print("\nAll EfficientNet benchmarks complete.")