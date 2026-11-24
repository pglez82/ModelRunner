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

# Queries nvidia-smi and returns (all_gpu_indexes, busy) where busy maps
# GPU index -> list of running processes. Returns None if nvidia-smi is unavailable.
def query_gpu_status():
    try:
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Advertencia: no se pudo consultar nvidia-smi; se omite la comprobación de GPU.")
        return None

    all_gpus = []
    uuid_to_index = {}
    for line in gpu_info.strip().splitlines():
        idx, uuid = [f.strip() for f in line.split(",")]
        all_gpus.append(idx)
        uuid_to_index[uuid] = idx

    busy = {}
    for line in apps.strip().splitlines():
        pid, name, mem, uuid = [f.strip() for f in line.split(",")]
        busy.setdefault(uuid_to_index.get(uuid, "?"), []).append(
            f"{name} (PID {pid}, {mem} MiB)")

    return all_gpus, busy

# Checks that the requested GPUs are free of running processes; exits with an error if not.
def check_gpus_free(gpus):
    status = query_gpu_status()
    if status is None:
        return
    _, busy = status

    for gpu in gpus.split(","):
        procs = busy.get(gpu.strip())
        if procs:
            print(f"Error: La GPU {gpu.strip()} está ocupada por: {', '.join(procs)}")
            sys.exit(1)


# For a single-GPU model whose GPU is busy, finds the first free GPU among the
# rest of the machine. Returns the fallback index or None if no free GPU exists.
def find_free_fallback_gpu(gpu, status):
    all_gpus, busy = status
    procs = busy.get(gpu)
    if not procs:
        return None
    print(f"Advertencia: La GPU {gpu} está ocupada por: {', '.join(procs)}")
    for candidate in all_gpus:
        if candidate != gpu and not busy.get(candidate):
            return candidate
    print("Error: No hay ninguna otra GPU libre para usar como alternativa.")
    return None


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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Model to launch directly: index (e.g. '1') or substring of the model name. "
             "Skips the interactive selection.",
    )
    parser.add_argument(
        "--gpu-fallback",
        action="store_true",
        help="If a single-GPU model has its GPU busy, run it on another free GPU "
             "instead of failing.",
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

    def select_model():
        print("\nModelos disponibles:\n")
        for i, m in enumerate(models, start=1):
            print(f"{i}) {m['name']}")

        choice = int(input("\nSelecciona modelo: "))
        return models[choice - 1]

    if args.model is None:
        cfg = select_model()
    else:
        if args.model.isdigit():
            idx = int(args.model) - 1
            if not 0 <= idx < len(models):
                sys.exit(f"Error: Índice de modelo inválido: {args.model} (1-{len(models)})")
            cfg = models[idx]
        else:
            matches = [m for m in models if args.model.lower() in m["name"].lower()]
            if not matches:
                sys.exit(f"Error: Ningún modelo coincide con: {args.model}")
            if len(matches) > 1:
                print("Varios modelos coinciden:")
                for m in matches:
                    print(f"  - {m['name']}")
                sys.exit("Error: Usa un término más específico o un índice.")
            cfg = matches[0]

    env = {**os.environ, **config.get("env", {})}
    if "gpus" in cfg:
        gpus = cfg["gpus"]
        if args.gpu_fallback and len(gpus.split(",")) == 1:
            status = query_gpu_status()
            if status is not None:
                fallback = find_free_fallback_gpu(gpus.strip(), status)
                if fallback is not None:
                    print(f"Lanzando el modelo en la GPU {fallback} en lugar de la GPU {gpus.strip()}.")
                    gpus = fallback
        check_gpus_free(gpus)
        env["CUDA_VISIBLE_DEVICES"] = gpus

    print(f"\nLanzando: {cfg['name']}\n")
    cmd = build_command(cfg, config)
    print(cmd)
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
