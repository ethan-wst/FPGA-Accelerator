# mobilenet_benchmark.py
# Run all MobileNet benchmarking scripts in the mobilenet subdirectory

import subprocess
import sys
import os
import torch

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), 'mobilenet')

# Check available devices
available_devices = ['cpu']
if torch.cuda.is_available():
    available_devices.append('cuda')
    print(f"CUDA is available. Will run benchmarks on both CPU and CUDA.")
else:
    print(f"CUDA is not available. Will run benchmarks on CPU only.")

# List of scripts and their model arguments
scripts_and_models = [
    # MobileNetV2
    ('mobilenetv2_benchmark.py', [
        'mobilenet_v2', 'mobilenet_v2_quant',
    ]),
    # MobileNetV3
    ('mobilenetv3_benchmark.py', [
        'mobilenet_v3_small', 'mobilenet_v3_large', 'mobilenet_v3_large_quant',
    ]),
]

# Run benchmarks on each device
for device in available_devices:
    print(f"\n===== Running benchmarks on {device.upper()} =====")
    
    for script, models in scripts_and_models:
        script_path = os.path.join(SCRIPT_DIR, script)
        for model in models:
            # Skip running quantized models on CUDA as they're designed for CPU
            if 'quant' in model and device == 'cuda':
                print(f"\nSkipping {model} on CUDA (quantized models run on CPU)")
                continue
                
            print(f"\n===== Running {script} for model {model} on {device} =====")
            cmd = [sys.executable, script_path, '--model', model, '--device', device]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Error running {script} for model {model} on {device}")
                continue

print("\nAll MobileNet benchmarks complete.")