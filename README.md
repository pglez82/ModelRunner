# ModelRunner

A lightweight CLI tool for easily launching `llama-server` (from `llama.cpp`) with pre-configured models. This tool simplifies the process of managing multiple LLM models, their specific configurations (like context length), and multimodal components.

## Features

- **Simple Menu-driven Interface**: Easily select and launch models from a list.
- **Configuration-driven**: All model settings and server paths are managed via JSON files.
- **Multimodal Support**: Automatically includes `mmproj` files for vision-capable models.
- **Customizable Parameters**: Set specific context lengths, ports, and GPU settings per model.
- **Environment Support**: Pass environment variables (like `CUDA_VISIBLE_DEVICES`) to the server process.

## Prerequisites

- **Python 3.x**
- **llama.cpp**: You must have `llama-server` installed and compiled.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ModelRunner
   ```

2. **Configure `config.json`:**
   Edit the `config.json` file to match your local environment. At a minimum, you need to specify:
   - `model_dir`: The directory where your `.gguf` files are located.
   - `llama_server`: The absolute path to your `llama-server` executable.
   - `port`: The default port to use.

   Example `config.json`:
   ```json
   {
       "model_dir": "/path/to/your/models",
       "llama_server": "/path/to/llama.cpp/build/bin/llama-server",
       "port": 8081,
       "default_context": 32768,
       "env": {
           "HSA_OVERRIDE_GFX_VERSION": "11.0.2"
       }
   }
   ```

3. **Configure `models.json`:**
   Add the models you want to use to `models.json`. Each entry can override default settings from `config.json`.

   Example `models.json` entry:
   ```json
   {
     "name": "Gemma 4 26B",
     "model": "gemma-4-26B-Q4_0.gguf",
     "mmproj": "mmproj-gemma-4-26B.gguf",
     "context": 128000,
     "port": 8081
   }
   ```

## Usage

Run the main script:

```bash
python3 run_model.py
```

A menu will appear in your terminal. Select the number corresponding to the model you wish to run, and the `llama-server` will be launched with the appropriate configuration.

## File Structure

- `run_model.py`: The main execution script.
- `config.json`: Global configuration (paths, defaults, environment).
- `models.json`: List of available models and their specific settings.
