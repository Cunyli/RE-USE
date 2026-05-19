import yaml
import torch
import os
import shutil
import glob
import re
from pathlib import Path
from torch.distributed import init_process_group


_ENV_DEFAULT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:\-([^}]*)\}")


def _expand_env_defaults(value):
    if not isinstance(value, str):
        return value

    def replace(match):
        name = match.group(1)
        default = match.group(2)
        return os.environ.get(name, default)

    return os.path.expanduser(os.path.expandvars(_ENV_DEFAULT_PATTERN.sub(replace, value)))


def _expand_paths(value):
    if isinstance(value, str):
        return _expand_env_defaults(value)
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    return value

def load_config(config_path):
    """Load configuration from a YAML file."""
    config_path = os.path.expanduser(os.path.expandvars(config_path))
    with open(config_path, 'r') as file:
        return _expand_paths(yaml.safe_load(file))

def initialize_seed(seed):
    """Initialize the random seed for both CPU and GPU."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def print_gpu_info(num_gpus, cfg):
    """Print information about available GPUs and batch size per GPU."""
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        print('Batch size per GPU:', int(cfg['training_cfg']['batch_size'] / num_gpus))

def initialize_process_group(cfg, rank):
    """Initialize the process group for distributed training."""
    init_process_group(
        backend=cfg['env_setting']['dist_cfg']['dist_backend'],
        init_method=cfg['env_setting']['dist_cfg']['dist_url'],
        world_size=cfg['env_setting']['dist_cfg']['world_size'] * cfg['env_setting']['num_gpus'],
        rank=rank
    )

def log_model_info(rank, model, exp_path):
    """Log model information and create necessary directories."""
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("Generator Parameters :", num_params)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'logs'), exist_ok=True)
    print("checkpoints directory :", exp_path)

def load_ckpts(args, device):
    """Load checkpoints if available."""
    if os.path.isdir(args.exp_path):
        cp_g = scan_checkpoint(args.exp_path, 'g_')
        cp_do = scan_checkpoint(args.exp_path, 'do_')
        if cp_g is None or cp_do is None:
            return None, None, 0, -1
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        return state_dict_g, state_dict_do, state_dict_do['steps'] + 1, state_dict_do['epoch']
    return None, None, 0, -1

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def prune_step_checkpoints(cp_dir, keep=3, prefixes=("g_", "do_")):
    cp_dir = Path(cp_dir)
    steps = []
    for path in cp_dir.glob(f"{prefixes[0]}*.pth"):
        try:
            steps.append(int(path.stem.replace(prefixes[0], "")))
        except ValueError:
            continue

    for step in sorted(steps)[:-keep]:
        for prefix in prefixes:
            checkpoint = cp_dir / f"{prefix}{step:08d}.pth"
            if checkpoint.exists():
                try:
                    checkpoint.unlink()
                    print(f"Deleted checkpoint: {checkpoint}")
                except OSError as exc:
                    print(f"Failed to delete checkpoint {checkpoint}: {exc}")


def save_best_step_checkpoints(cp_dir, step, generator_state, optimizer_state, generator_prefix="g_", optimizer_prefix="do_"):
    cp_dir = Path(cp_dir)
    save_checkpoint(str(cp_dir / f"best_{generator_prefix}{step:08d}.pth"), generator_state)
    save_checkpoint(str(cp_dir / f"best_{optimizer_prefix}{step:08d}.pth"), optimizer_state)

    for pattern in (f"best_{generator_prefix}*.pth", f"best_{optimizer_prefix}*.pth"):
        for checkpoint in cp_dir.glob(pattern):
            if f"{step:08d}" not in checkpoint.name:
                checkpoint.unlink()


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????' + '.pth')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def build_env(config, config_name, exp_path):
    os.makedirs(exp_path, exist_ok=True)
    t_path = os.path.join(exp_path, config_name)
    if config != t_path:
        shutil.copyfile(config, t_path)

def load_optimizer_states(optimizers, state_dict_do):
    """Load optimizer states from checkpoint."""
    if state_dict_do is not None:
        optim_g, optim_d = optimizers
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
