#!/usr/bin/env python3

import json
import subprocess
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"
MODELS_FILE = Path(__file__).parent / "models.json"

# Loads the global configuration from the config.json file.
def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)
    
# Builds the command to run the llama server based on the model configuration and global configuration.
def build_command(cfg, config):
    model_dir = Path(config["model_dir"])
    default_ctx = config.get("default_context", 32768)
    default_n_cpu_moe = config.get("default_n_cpu_moe", 0)
    port = cfg.get("port", config["port"])

    cmd = [
        config["llama_server"],
        "-ngl", "999",
        "-np", "1",
        "--no-mmap",
        "--port", str(port),
        "--n-cpu-moe", str(cfg.get("n_cpu_moe", default_n_cpu_moe)),
        "-c", str(cfg.get("context", default_ctx)),
        "-m", str(model_dir / cfg["model"]),
        "-fa","on",
        "-ctk", "q8_0",
        "-ctv", "q8_0"
    ]

    if "mmproj" in cfg:
        cmd += ["--mmproj", str(model_dir / cfg["mmproj"])]
    if "model-draft" in cfg:
        cmd += ["--model-draft", str(model_dir / cfg["model-draft"])]


    if cfg.get("mtp", False) or "model-draft" in cfg:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]

    return cmd


def main():
    config = load_config()

    with open(MODELS_FILE, "r") as f:
        models = json.load(f)

    print("\nModelos disponibles:\n")

    for i, m in enumerate(models, start=1):
        print(f"{i}) {m['name']}")

    choice = int(input("\nSelecciona modelo: "))
    cfg = models[choice - 1]

    env = {**config.get("env", {})}
    if "gpus" in cfg:
        env["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]

    print(f"\nLanzando: {cfg['name']}\n")
    cmd = build_command(cfg, config)
    print(cmd)
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
