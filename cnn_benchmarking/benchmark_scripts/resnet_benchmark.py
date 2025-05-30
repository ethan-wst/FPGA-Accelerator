# resnet_benchmark.py
# Run all ResNet benchmarking scripts in the resnet subdirectory

import subprocess
import sys
import os

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), 'resnet')

# List of scripts and their model arguments
scripts_and_models = [
    # ResNet models
    ('resnet18_benchmark.py', [
        'resnet18'
    ]),
    ('resnet50_benchmark.py', [
        'resnet50'
    ]),
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

print("\nAll ResNet benchmarks complete.")