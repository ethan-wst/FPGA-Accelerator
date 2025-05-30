# onnx_benchmark.py
# Run all ONNX benchmarking scripts across different model families with both standard and quantized variants

import subprocess
import sys
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Run all ONNX benchmarks')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                   help='Device to run benchmarks on')
parser.add_argument('--skip_quant', action='store_true',
                   help='Skip quantized models (only run standard models)')
args = parser.parse_args()

# Base directories
SCRIPT_DIR = os.path.dirname(__file__)

# List of model families, their scripts, and model variants
model_families = [
    # MobileNet family
    {
        'dir': 'mobilenet',
        'scripts': [
            {
                'name': 'mobilenetv2_onnx_benchmark.py',
                'models': ['mobilenet_v2', 'mobilenet_v2_quant']
            }
        ]
    },
    
    # ResNet family
    {
        'dir': 'resnet',
        'scripts': [
            {
                'name': 'resnet50_onnx_benchmark.py',
                'models': ['resnet50', 'resnet50_quant']
            }
        ]
    },
    
    # EfficientNet family
    {
        'dir': 'efficientnet',
        'scripts': [
            {
                'name': 'efficientnet_lite4_onnx_benchmark.py',
                'models': ['efficientnet_lite4', 'efficientnet_lite4_quant']
            }
        ]
    }
]

# Filter out quantized models if skip_quant is True
if args.skip_quant:
    for family in model_families:
        for script in family['scripts']:
            script['models'] = [model for model in script['models'] if not model.endswith('_quant')]

# Count total benchmarks
total_benchmarks = sum(len(script['models']) for family in model_families for script in family['scripts'])
completed_benchmarks = 0

print(f"Starting ONNX benchmarking on {args.device}")
print(f"Total benchmarks to run: {total_benchmarks}")
print("=" * 80)

# Run all benchmarks
for family in model_families:
    family_dir = os.path.join(SCRIPT_DIR, family['dir'])
    
    for script_info in family['scripts']:
        script_path = os.path.join(family_dir, script_info['name'])
        
        # Skip if script doesn't exist
        if not os.path.exists(script_path):
            print(f"Warning: Script {script_path} not found, skipping.")
            continue
            
        for model in script_info['models']:
            completed_benchmarks += 1
            print(f"\n[{completed_benchmarks}/{total_benchmarks}] Running {script_info['name']} for model {model} on {args.device}")
            print("-" * 80)
            
            cmd = [sys.executable, script_path, '--model', model, '--device', args.device]
            
            try:
                result = subprocess.run(cmd, check=True)
                print(f"✓ Successfully completed {model}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Error running {script_info['name']} for model {model}")
                print(f"  Return code: {e.returncode}")
            except Exception as e:
                print(f"✗ Unexpected error running {script_info['name']} for model {model}")
                print(f"  Error: {str(e)}")

print("\n" + "=" * 80)
print(f"ONNX benchmarks completed: {completed_benchmarks}/{total_benchmarks}")
print("Results saved to: /home/ethanwst/FPGA-Accelerator/cnn_benchmarking/benchmark_results/onnx_local_benchmark_results.csv")