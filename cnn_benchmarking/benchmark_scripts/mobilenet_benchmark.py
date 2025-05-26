# mobilenet_benchmark.py
# Run all MobileNet benchmarking scripts in the mobilenet subdirectory

import subprocess
import sys
import os

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), 'mobilenet')

# List of scripts and their model arguments
scripts_and_models = [
    # MobileNetV2
    ('mobilenetv2_benchmark.py', [
        'mobilenet_v2',
    ]),
    # MobileNetV3
    ('mobilenetv3_benchmark.py', [
        'mobilenet_v3_small', 'mobilenet_v3_large',
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
print("\nAll MobileNet benchmarks complete.")
