# speedrun

Simple tool to run ML training on vast.ai GPU instances.

## Quick Start

1. **Get your API key from https://vast.ai/console/api-keys and set it:**
   ```bash
   pip install vastai
   vastai set api-key YOUR_API_KEY
   ```

2. **Run training:**
   ```bash
   ./speedrun.sh ./your_project
   ```

That's it! The script automatically sets up everything and runs your training on powerful GPUs.

## What it does

- Finds the most powerful available GPU on vast.ai
- Uploads your project and runs `train.py`
- Downloads model artifacts when complete
- Handles instance cleanup and cost tracking

## Requirements

- Python 3.8+
- Project with `train.py` entry point