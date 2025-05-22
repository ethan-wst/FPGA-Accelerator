# FPGA-Accelerator

## Overview
This repository provides scripts and tools for benchmarking CNN models (MobileNetV2, MobileNetV3, EfficientNet) on a subset of the ImageNet validation dataset. It includes utilities for running benchmarks, collecting results, and visualizing inference speed and accuracy.

## Directory Structure

- `cnn_benchmarking/`
  - `requirements.txt` — Python dependencies for all scripts
  - `benchmark_scripts/` — Main benchmarking and plotting scripts
    - `mobilenetv2_benchmark.py` — Benchmark MobileNetV2
    - `mobilenetv3_benchmark.py` — Benchmark MobileNetV3
    - `efficientnet_benchmark.py` — Benchmark EfficientNet (timm)
    - `plot_benchmarks.py` — Visualize benchmark results (inference time, top-1/top-5 accuracy)
  - `benchmark_results/` — Output directory for benchmark results and plots
    - `benchmark_results.csv` — Results CSV (auto-generated)
    - `benchmark_plot.png` — Plot of results (auto-generated)
  - `imagenet/` — ImageNet-related data (not tracked in git)
    - `imagenet_subset/` — Subset of ImageNet validation images, organized by WNID (not tracked in git)
    - `imagenet_class_index.json` — Official ImageNet class index mapping (required for accuracy)
    - `ILSVRC2012_devkit_t12/` — ImageNet devkit (ground truth, meta, etc.)
      - `data/ILSVRC2012_validation_ground_truth.txt` — Validation ground truth
      - `data/meta.mat` — Class meta info

## Getting Started

1. **Install dependencies:**
   ```sh
   pip install -r cnn_benchmarking/requirements.txt
   ```

2. **Prepare ImageNet subset:**
   - Place your subset of ImageNet validation images in `cnn_benchmarking/imagenet/imagenet_subset/`, organized by WNID (folder per class).
   - Place `imagenet_class_index.json` in `cnn_benchmarking/imagenet/` (download from torchvision repo if needed).
   - Place the devkit in `cnn_benchmarking/imagenet/ILSVRC2012_devkit_t12/`.

3. **Run a benchmark:**
   ```sh
   cd cnn_benchmarking/benchmark_scripts
   python mobilenetv2_benchmark.py
   python mobilenetv3_benchmark.py
   python efficientnet_benchmark.py
   ```
   - Results will be saved to `../benchmark_results/benchmark_results.csv`.

4. **Visualize results:**
   ```sh
   python plot_benchmarks.py
   ```
   - This will generate `benchmark_plot.png` in `../benchmark_results/`.

## Notes
- The full ImageNet dataset and large image folders are not tracked in git (see `.gitignore`).
- All scripts are designed to be run from the `cnn_benchmarking/benchmark_scripts/` directory.
- For best results, use Python 3.8+ and the package versions in `requirements.txt`.

## License
See `cnn_benchmarking/imagenet/ILSVRC2012_devkit_t12/COPYING` for ImageNet devkit license.