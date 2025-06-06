# speedrun

Simple tool to run ML training on vast.ai GPU instances.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install vastai
   ```

2. **Set up vast.ai API key:**
   ```bash
   vastai set api-key YOUR_API_KEY
   ```

3. **Run training:**
   ```bash
   # Test first (no costs)
   python speedrun.py ./your_project --dry-run
   
   # Run actual training
   python speedrun.py ./your_project
   ```

## Requirements

- Python 3.8+
- vast.ai API key from https://vast.ai/console/api-keys
- Project with `train.py` entry point

## Features

- Automatically finds the most powerful available GPU
- Uploads your project and runs training
- Downloads model artifacts when complete
- Handles instance cleanup and cost tracking

## License

MIT