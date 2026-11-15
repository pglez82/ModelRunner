#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Loads the global configuration from the specified config file.
def load_config(config_file):
    if not config_file.exists():
        sys.exit(f"Error: Configuration file not found: {config_file}")
    with open(config_file, "r") as f:
        return json.load(f)

# Loads the models list from the specified models file.
def load_models(models_file):
    if not models_file.exists():
        sys.exit(f"Error: Models file not found: {models_file}")
    with open(models_file, "r") as f:
        return json.load(f)

# Builds the command to run the llama server based on the model configuration and global configuration.
def build_command(cfg, config):
    model_dir = Path(config["model_dir"])
    default_ctx = config.get("default_context", 32768)
    default_n_cpu_moe = config.get("default_n_cpu_moe", 0)
    port = cfg.get("port", config["port"])
    host = cfg.get("host", config.get("host", "0.0.0.0"))
    ngl = cfg.get("ngl", config.get("default_ngl", "999"))
    ctk = cfg.get("ctk", config.get("default_ctk", "q8_0"))
    ctv = cfg.get("ctv", config.get("default_ctv", "q8_0"))
    load_mode = cfg.get("load_mode", config.get("load_mode", "auto"))

    cmd = [
        config["llama_server"],
        "-ngl", str(ngl),
        "-np", "1",
        "--load-mode", str(load_mode),
        "--host", str(host),
        "--port", str(port),
        "--n-cpu-moe", str(cfg.get("n_cpu_moe", default_n_cpu_moe)),
        "-c", str(cfg.get("context", default_ctx)),
        "-m", str(model_dir / cfg["model"]),
        "-fa", "on",
        "-ctk", str(ctk),
        "-ctv", str(ctv)
    ]

    threads = cfg.get("threads", config.get("threads"))
    if threads is not None:
        cmd += ["-t", str(threads)]

    threads_batch = cfg.get("threads_batch", config.get("threads_batch"))
    if threads_batch is not None:
        cmd += ["-tb", str(threads_batch)]

    ubatch = cfg.get("ubatch_size", config.get("ubatch_size"))
    if ubatch is not None:
        cmd += ["-ub", str(ubatch)]

    if "spec_draft_n_cpu_moe" in cfg:
        cmd += ["--spec-draft-n-cpu-moe", str(cfg["spec_draft_n_cpu_moe"])]

    if "mmproj" in cfg:
        cmd += ["--mmproj", str(model_dir / cfg["mmproj"])]
    if "model-draft" in cfg:
        cmd += ["--model-draft", str(model_dir / cfg["model-draft"])]

    if cfg.get("mtp", False) or "model-draft" in cfg:
        spec_n_max = cfg.get("spec_draft_n_max", 3)
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(spec_n_max)]

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run llama-server with pre-configured models.")
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="Machine identifier to load config-<machine>.json and models-<machine>.json",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    machine = args.machine
    if not machine:
        if (base_dir / "config.json").exists() and (base_dir / "models.json").exists():
            config_file = base_dir / "config.json"
            models_file = base_dir / "models.json"
        else:
            available_machines = sorted([
                p.stem.replace("models-", "")
                for p in base_dir.glob("models-*.json")
                if (base_dir / f"config-{p.stem.replace('models-', '')}.json").exists()
            ])
            if len(available_machines) == 1:
                machine = available_machines[0]
                config_file = base_dir / f"config-{machine}.json"
                models_file = base_dir / f"models-{machine}.json"
            elif available_machines:
                print("\nMáquinas disponibles:\n")
                for i, m in enumerate(available_machines, start=1):
                    print(f"{i}) {m}")
                choice = int(input("\nSelecciona máquina: "))
                machine = available_machines[choice - 1]
                config_file = base_dir / f"config-{machine}.json"
                models_file = base_dir / f"models-{machine}.json"
            else:
                config_file = base_dir / "config.json"
                models_file = base_dir / "models.json"
    else:
        config_file = base_dir / f"config-{machine}.json"
        models_file = base_dir / f"models-{machine}.json"

    config = load_config(config_file)
    models = load_models(models_file)

    print("\nModelos disponibles:\n")

    for i, m in enumerate(models, start=1):
        print(f"{i}) {m['name']}")

    choice = int(input("\nSelecciona modelo: "))
    cfg = models[choice - 1]

    env = {**os.environ, **config.get("env", {})}
    if "gpus" in cfg:
        env["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]

    print(f"\nLanzando: {cfg['name']}\n")
    cmd = build_command(cfg, config)
    print(cmd)
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
