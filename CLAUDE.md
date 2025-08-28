# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOvision Pro is a comprehensive object detection development toolkit focused on small object detection optimization, particularly for drone scenarios. It implements the Drone-YOLO algorithm and provides a complete workflow from data annotation to model training and deployment.

## Key Architecture

### Core Components

1. **Drone-YOLO Algorithm** - Enhanced YOLOv8 for small object detection
   - RepVGGBlock backbone for efficiency
   - P2 detection head for 4-16 pixel small objects
   - Sandwich fusion structure for cross-scale feature integration

2. **PyQt5 UI System** (`main.py`) - Main application interface
   - Multi-source input support (image/video/camera)
   - Real-time parameter adjustment
   - Dual-view comparison display

3. **Modular Script System** (`scripts/`) - Organized by functionality
   - Data processing tools
   - Validation and checking utilities
   - Model testing and visualization

### Directory Structure

```
Vision-Pro/
├── main.py                    # PyQt5 UI application
├── train.py                   # Model training script
├── scripts/                   # Modular script system
│   ├── data_processing/       # Data conversion and preprocessing
│   │   ├── visdrone/         # VisDrone2019 dataset tools
│   │   └── general/          # General data processing
│   ├── validation/           # Environment and data validation
│   ├── demo/                 # Feature demonstrations
│   ├── testing/              # Model testing
│   └── visualization/        # Architecture and results visualization
├── assets/configs/           # Model configurations
├── data/                     # Datasets and annotations
├── models/                   # Pretrained models
├── outputs/                  # Training outputs and logs
└── ultralytics/              # Modified YOLOv8 framework
```

## Common Development Commands

### Environment Setup and Validation

```bash
# Quick environment check (recommended first)
python scripts/validation/simple_check.py

# Complete environment validation
python scripts/validation/verify_local_ultralytics.py

# Run all verification checks
python scripts/run.py check
```

### Data Processing (VisDrone)

```bash
# One-click VisDrone processing (recommended)
python scripts/data_processing/visdrone/process_visdrone_complete.py \
    --input data/VisDrone2019-DET-train \
    --output data/visdrone_yolo \
    --verbose

# Step-by-step processing
python scripts/data_processing/visdrone/convert_visdrone.py \
    -i data/VisDrone2019-DET-train -o data/visdrone_yolo
python scripts/data_processing/visdrone/split_visdrone_dataset.py \
    -i data/visdrone_yolo -o data/visdrone_yolo
python scripts/data_processing/visdrone/validate_visdrone_dataset.py \
    -d data/visdrone_yolo --visualize
```

### Model Training

```bash
# Standard YOLOv8 training
yolo train data=data/visdrone_yolo/data.yaml model=yolov8s.pt epochs=100 imgsz=640

# Drone-YOLO training
python train.py

# Custom configuration training
yolo train data=data/visdrone_yolo/data.yaml model=assets/configs/yolov8s-drone.yaml epochs=300
```

### Testing and Development

```bash
# Test Drone-YOLO model
python scripts/testing/test_drone_yolo.py

# Generate architecture visualization
python scripts/visualization/visualize_drone_yolo.py

# Run core concept demonstrations
python scripts/demo/drone_yolo_demo.py

# Unified tool system (recommended)
python scripts/yolo_tools.py <command> <subcommand> [options]
python scripts/run.py                    # Interactive menu
python scripts/quick_commands.py <task> # Quick commands
```

### Running the UI Application

```bash
# Launch the main application
python main.py
```

## Important Technical Details

### Drone-YOLO Configuration

- **Model Architecture**: Defined in `assets/configs/yolov8s-drone.yaml`
- **Key Features**: 
  - RepVGGBlock replaces standard Conv layers
  - P2 detection head (160×160 feature map)
  - 4 detection scales: P2, P3, P4, P5
- **Performance**: 11.1M parameters, 40.3 GFLOPs

### VisDrone Dataset Processing

- **Class Mapping**: 10 VisDrone classes mapped to YOLO format
- **Data Split**: 8:1:1 train/validation/test ratio
- **Format Conversion**: VisDrone annotation format to YOLO format
- **Output**: Processed data in `data/visdrone_yolo/`

### Dependencies

Based on the embedded ultralytics framework:
- Python >= 3.8
- PyTorch >= 1.8.0 (except 2.4.0 on Windows)
- OpenCV >= 4.6.0
- PyQt5 (for UI application)
- NumPy, matplotlib, pillow, pyyaml, scipy, requests

## Development Guidelines

### Code Standards

- Follow the existing code style and structure
- Include detailed docstrings and Chinese comments
- Use argparse for command-line scripts
- Implement proper error handling and logging
- Maintain the modular organization of scripts

### Script Organization

- **data_processing/**: Dataset conversion and preprocessing
- **validation/**: Environment checking and data validation
- **testing/**: Model functionality testing
- **demo/**: Feature demonstrations
- **visualization/**: Results and architecture visualization

### Model Development

- Model configurations in `assets/configs/`
- Use the unified tool system for consistent operations
- Test with validation scripts before training
- Document new features in appropriate README files

## Common Issues and Solutions

### Environment Setup

- Use `simple_check.py` for basic environment validation
- Run `verify_local_ultralytics.py` for detailed configuration analysis
- Check GPU availability and CUDA installation

### Data Processing

- Ensure VisDrone dataset follows expected directory structure
- Validate annotation format before conversion
- Use the one-click processing script for reliable results

### Model Training

- Verify dataset configuration before training
- Monitor training progress through outputs/logs/
- Test trained models using the validation scripts

## Testing and Validation

### Environment Tests
```bash
python scripts/validation/simple_check.py
python scripts/validation/verify_local_ultralytics.py
```

### Data Tests
```bash
python scripts/data_processing/visdrone/validate_visdrone_dataset.py
python scripts/validation/test_visdrone_conversion.py
```

### Model Tests
```bash
python scripts/testing/test_drone_yolo.py
python scripts/demo/drone_yolo_demo.py
```

## Project-Specific Notes

- The project uses a modified ultralytics framework in the `ultralytics/` directory
- VisDrone2019 dataset support is a core feature with specialized processing tools
- The Drone-YOLO implementation includes architectural improvements for small object detection
- All scripts support both individual execution and unified tool system usage

用中文对话