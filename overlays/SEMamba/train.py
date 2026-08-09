import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import warnings

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from joblib import Parallel, delayed
from pesq import pesq
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

warnings.simplefilter(action='ignore', category=FutureWarning)

from dataloaders.dataloader_vctk import VCTKDemandDataset
from dataloaders.dataloader_use_simulation import USESimulationPairDataset
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.loss import phase_losses
from models.discriminator import MetricDiscriminator, batch_pesq
from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed,
    print_gpu_info, log_model_info, initialize_process_group,
    prune_step_checkpoints, save_best_step_checkpoints,
)
VALIDATION_EVALUATION_DIR = os.environ.get(
    "USE_VALIDATION_EVALUATION_DIR",
    "/scratch/work/lil14/use_simulation_pipeline/scripts/evaluation",
)
if VALIDATION_EVALUATION_DIR not in sys.path:
    sys.path.insert(0, VALIDATION_EVALUATION_DIR)
from validation_avqi_gap import format_wandb_avqi_metrics, run_validation_avqi_metrics

try:
    import wandb
except ImportError:
    wandb = None

torch.backends.cudnn.benchmark = True

WANDB_STANDARD_TAG = "wandb_standard_v1"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed, rank=0, offset=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(rank) * 1000 + int(offset))
    return generator


def scalar_value(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def add_metric_window(metric_sums, metrics):
    for name, value in metrics.items():
        metric_sums[name] = metric_sums.get(name, 0.0) + scalar_value(value)


def average_metric_window(metric_sums, count):
    if count <= 0:
        return {}
    return {name: value / count for name, value in metric_sums.items()}


def name_token(value, default="na"):
    text = str(value if value not in (None, "") else default).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or default


def build_wandb_identity(repo_name, model_name, dataset_type, experiment, change, timestamp):
    run_name = "__".join(
        [
            name_token(timestamp),
            name_token(repo_name),
            name_token(model_name),
            name_token(dataset_type),
            name_token(change or experiment),
        ]
    )
    group = "__".join(
        [
            name_token(repo_name),
            name_token(model_name),
            name_token(dataset_type),
            name_token(experiment),
        ]
    )
    return run_name, group


def setup_optimizers(models, cfg):
    """Set up optimizers for the models."""
    generator, discriminator = models
    learning_rate = cfg['training_cfg']['learning_rate']
    betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])

    optim_g = optim.AdamW(generator.parameters(), lr=learning_rate, betas=betas)
    optim_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=betas)

    return optim_g, optim_d

def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d


def log_validation_avqi_metrics(generator, cfg, device, global_step):
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    was_training = generator.training
    generator.eval()
    try:
        @torch.inference_mode()
        def enhance_one(path):
            noisy, _ = librosa.load(path, sr=sampling_rate)
            noisy = torch.as_tensor(noisy, dtype=torch.float32, device=device)
            norm = torch.sqrt(noisy.numel() / torch.sum(noisy ** 2.0).clamp_min(1e-12))
            noisy = (noisy * norm).unsqueeze(0)
            mag, pha, _ = mag_phase_stft(noisy, n_fft, hop_size, win_size, compress_factor)
            enhanced_mag, enhanced_pha, _ = generator(mag, pha)
            enhanced = mag_phase_istft(enhanced_mag, enhanced_pha, n_fft, hop_size, win_size, compress_factor)
            return (enhanced / norm).squeeze().cpu().numpy(), sampling_rate

        return run_validation_avqi_metrics(
            "reuse",
            global_step,
            enhance_one,
            pair_csv=cfg['data_cfg']['valid_pair_manifest'],
        )
    finally:
        generator.train(was_training)


def save_best_avqi_gap_checkpoints(generator, discriminator, optimizers, exp_path, epoch, global_step, num_gpus, gap):
    generator_module = generator.module if num_gpus > 1 else generator
    best_gap = getattr(generator_module, "_best_avqi_gap_to_clean", float("inf"))
    if gap < 0 or gap >= best_gap:
        return

    optim_g, optim_d = optimizers
    generator_module._best_avqi_gap_to_clean = gap
    generator_state = {"generator": generator_module.state_dict()}
    optimizer_state = {
        "discriminator": (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "steps": global_step,
        "epoch": epoch,
        "best_avqi_gap_to_clean": gap,
    }
    save_best_step_checkpoints(
        exp_path,
        global_step,
        generator_state,
        optimizer_state,
        generator_prefix="avqi_gap_g_",
        optimizer_prefix="avqi_gap_do_",
    )


def save_best_guarded_checkpoints(generator, discriminator, optimizers, exp_path, epoch, global_step, num_gpus, val_loss):
    generator_module = generator.module if num_gpus > 1 else generator
    latest_gap = getattr(generator_module, "_latest_avqi_gap_to_clean", None)
    best_loss = getattr(generator_module, "_best_guarded_val_loss", float("inf"))
    if latest_gap is None or latest_gap < 0 or val_loss >= best_loss:
        return

    optim_g, optim_d = optimizers
    generator_module._best_guarded_val_loss = val_loss
    generator_state = {"generator": generator_module.state_dict()}
    optimizer_state = {
        "discriminator": (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "steps": global_step,
        "epoch": epoch,
        "best_avqi_gap_to_clean": getattr(generator_module, "_best_avqi_gap_to_clean", float("inf")),
        "latest_avqi_gap_to_clean": latest_gap,
        "best_guarded_val_loss": val_loss,
    }
    save_best_step_checkpoints(
        exp_path,
        global_step,
        generator_state,
        optimizer_state,
        generator_prefix="guarded_g_",
        optimizer_prefix="guarded_do_",
    )


def create_dataset(cfg, train=True, split=True, device='cuda:0'):
    """Create dataset based on cfguration."""
    if cfg['data_cfg'].get('dataset_type') == 'use_simulation_fixed':
        pair_manifest = cfg['data_cfg']['train_pair_manifest'] if train else cfg['data_cfg']['valid_pair_manifest']
        return USESimulationPairDataset(
            pair_manifest=pair_manifest,
            use_simulation_root=cfg['data_cfg'].get('use_simulation_root', '../USE_simulation'),
            sampling_rate=cfg['stft_cfg']['sampling_rate'],
            segment_size=cfg['training_cfg']['segment_size'],
            n_fft=cfg['stft_cfg']['n_fft'],
            hop_size=cfg['stft_cfg']['hop_size'],
            win_size=cfg['stft_cfg']['win_size'],
            compress_factor=cfg['model_cfg']['compress_factor'],
            split=split,
            random_start=train,
            normalize=True,
            pcs=cfg['training_cfg']['use_PCS400'] if train else False,
            seed=cfg['env_setting']['seed'],
            mode='train' if train else 'validation',
            return_metadata=not train,
        )

    clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    noisy_json = cfg['data_cfg']['train_noisy_json'] if train else cfg['data_cfg']['valid_noisy_json']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False

    return VCTKDemandDataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs
    )

def create_dataloader(dataset, cfg, train=True, rank=0):
    """Create dataloader based on dataset and configuration."""
    if train and cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset, seed=int(cfg['env_setting']['seed']))
        batch_size = cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True if train else False,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(cfg['env_setting']['seed'], rank=rank, offset=0 if train else 1),
    )


def metadata_value(metadata, key, index):
    if not metadata:
        return ""
    if isinstance(metadata, dict):
        value = metadata.get(key, "")
    elif isinstance(metadata, (list, tuple)) and index < len(metadata) and isinstance(metadata[index], dict):
        value = metadata[index].get(key, "")
    else:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
        return value[index] if index < len(value) else ""
    if torch.is_tensor(value):
        return value[index].item() if value.ndim > 0 else value.item()
    return value


def load_pesq_invalid_ids(cfg):
    path = cfg.get("data_cfg", {}).get("pesq_invalid_manifest", "")
    if not path:
        return set()
    invalid_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("uid") or record.get("id")
            if sample_id:
                invalid_ids.add(sample_id)
    print(f"Loaded {len(invalid_ids)} PESQ-invalid sample ids from {path}")
    return invalid_ids


def pesq_score_with_details(utts_r, utts_g, cfg, metadata=None, invalid_ids=None, limit=None):
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    invalid_ids = invalid_ids or set()
    eligible = [
        i for i in range(len(utts_r))
        if str(metadata_value(metadata, "id", i)) not in invalid_ids
    ]
    if limit is not None:
        eligible = eligible[:limit]
    skipped = len(utts_r) - len(eligible)
    num_workers = max(1, min(int(cfg['env_setting'].get('num_workers', 1)), len(utts_r)))

    def eval_pesq(index, clean_utt, esti_utt):
        clean_np = clean_utt.squeeze().cpu().numpy()
        esti_np = esti_utt.squeeze().cpu().numpy()
        try:
            score = float(pesq(sampling_rate, clean_np, esti_np, 'wb'))
        except Exception as exc:
            return index, None, str(exc)
        if score < 0:
            return index, None, "negative PESQ score"
        return index, score, ""

    results = Parallel(n_jobs=num_workers)(
        delayed(eval_pesq)(i, utts_r[i], utts_g[i]) for i in eligible
    )

    valid_scores = []
    invalid = []
    for index, score, error in results:
        if score is None:
            invalid.append({
                "index": int(index),
                "id": str(metadata_value(metadata, "id", index)),
                "clean_path": str(metadata_value(metadata, "clean_path", index)),
                "noisy_path": str(metadata_value(metadata, "noisy_path", index)),
                "error": error,
            })
        else:
            valid_scores.append(score)

    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
    return mean_score, len(valid_scores), invalid, skipped


def log_invalid_pesq_samples(exp_path, step, invalid, limit):
    if not invalid:
        return
    log_path = os.path.join(exp_path, "pesq_invalid_samples.jsonl")
    with open(log_path, "a") as handle:
        for item in invalid:
            record = {"step": int(step), **item}
            handle.write(json.dumps(record) + "\n")
    for item in invalid[:limit]:
        print(
            "Invalid PESQ sample at step {step}: id={id}, clean={clean}, noisy={noisy}, error={error}".format(
                step=step,
                id=item.get("id", ""),
                clean=item.get("clean_path", ""),
                noisy=item.get("noisy_path", ""),
                error=item.get("error", ""),
            )
        )
    if len(invalid) > limit:
        print(f"Invalid PESQ samples at step {step}: {len(invalid) - limit} more written to {log_path}")


def train(rank, args, cfg):
    num_gpus = cfg['env_setting']['num_gpus']
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    batch_size = cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']
    if num_gpus >= 1:
        initialize_process_group(cfg, rank)
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    generator = SEMamba(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)

    if rank == 0:
        log_model_info(rank, generator, args.exp_path)

    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts(args, device)
    if state_dict_g is not None:
        generator.load_state_dict(state_dict_g['generator'], strict=False)
        discriminator.load_state_dict(state_dict_do['discriminator'], strict=False)
    generator._best_avqi_gap_to_clean = (
        float(state_dict_do.get("best_avqi_gap_to_clean", float("inf")))
        if state_dict_do is not None
        else float("inf")
    )
    generator._latest_avqi_gap_to_clean = (
        float(state_dict_do["latest_avqi_gap_to_clean"])
        if state_dict_do is not None and "latest_avqi_gap_to_clean" in state_dict_do
        else None
    )
    generator._best_guarded_val_loss = (
        float(state_dict_do.get("best_guarded_val_loss", float("inf")))
        if state_dict_do is not None
        else float("inf")
    )

    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    if cfg['training_cfg'].get('use_pretrainedD', False):
        pretrained_d_path = Path(
            cfg['training_cfg'].get(
                'pretrained_discriminator_path',
                Path(__file__).resolve().parents[2] / 'pretrained/semamba/pretrained_discriminator.pth',
            )
        ).expanduser()
        discriminator.load_state_dict(torch.load(pretrained_d_path, map_location=device))
        print(f"Loaded pretrained discriminator from {pretrained_d_path}.")

    # Create optimizer and schedulers
    optimizers = setup_optimizers((generator, discriminator), cfg)
    load_optimizer_states(optimizers, state_dict_do)
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = setup_schedulers(optimizers, cfg, last_epoch)

    # Create trainset and train_loader
    trainset = create_dataset(cfg, train=True, split=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True, rank=rank)

    # Create validset and validation_loader if rank is 0
    if rank == 0:
        validset = create_dataset(cfg, train=False, split=False, device=device)
        validation_loader = create_dataloader(validset, cfg, train=False, rank=rank)
        sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))
        wandb_run = None
        wandb_cfg = cfg.get('wandb_cfg', {})
        if wandb_cfg.get('use_wandb', False):
            if wandb is None:
                raise RuntimeError("wandb_cfg.use_wandb=true but wandb is not installed.")
            repo_name = cfg.get('repo_name', wandb_cfg.get('repo_name', 'reuse'))
            model_name = cfg.get('model_name', wandb_cfg.get('model_name', 'semamba'))
            dataset_type = cfg.get('data_cfg', {}).get('dataset_type', 'unknown_dataset')
            experiment = cfg.get('experiment', wandb_cfg.get('experiment', args.exp_name))
            change = wandb_cfg.get('change') or cfg.get('wandb_change') or cfg.get('change') or experiment
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            default_run_name, default_group = build_wandb_identity(
                repo_name,
                model_name,
                dataset_type,
                experiment,
                change,
                timestamp,
            )
            wandb_init_kwargs = {
                "project": wandb_cfg.get('project', 'reuse'),
                "entity": wandb_cfg.get('entity'),
                "name": wandb_cfg.get('run_name') or default_run_name,
                "group": wandb_cfg.get('group') or default_group,
                "mode": wandb_cfg.get('mode', 'online'),
                "tags": list(dict.fromkeys(wandb_cfg.get('tags', []) + [WANDB_STANDARD_TAG])),
                "config": {
                    **cfg,
                    "repo_name": repo_name,
                    "model_name": model_name,
                    "experiment": experiment,
                    "wandb_change": change,
                    "wandb_group": wandb_cfg.get('group') or default_group,
                },
            }
            if wandb_cfg.get('run_id'):
                wandb_init_kwargs["id"] = wandb_cfg['run_id']
                wandb_init_kwargs["resume"] = wandb_cfg.get('resume', 'allow')
            wandb_run = wandb.init(**wandb_init_kwargs)
            wandb.define_metric("charts/global_step", overwrite=True)
            wandb.define_metric("*", step_metric="charts/global_step", step_sync=True, overwrite=True)
            wandb.define_metric("trainer/global_step", hidden=True, overwrite=True)
            wandb.define_metric("charts/epoch", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("train/*", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("val/*", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("val_avqi/*", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("val_avqi_pathology/*", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("val_avqi_health/*", step_metric="charts/global_step", overwrite=True)
            wandb.define_metric("charts/*", step_metric="charts/global_step", overwrite=True)
    else:
        wandb_run = None

    generator.train()
    discriminator.train()

    train_metric_sums = {}
    train_metric_count = 0
    train_window_elapsed_sec = 0.0
    samples_seen = 0
    train_window_samples = 0
    summary_interval = max(1, int(cfg['env_setting'].get('wandb_log_interval_steps', cfg['env_setting'].get('summary_interval', 100))))
    validation_interval = max(1, int(cfg['env_setting'].get('validation_interval_steps', cfg['env_setting'].get('validation_interval', 500))))
    checkpoint_interval = max(1, int(cfg['env_setting'].get('checkpoint_interval_steps', cfg['env_setting'].get('checkpoint_interval', 1000))))
    avqi_interval = max(1, int(cfg['env_setting'].get('avqi_validation_interval_steps', 1000)))
    use_metric_loss = bool(cfg['training_cfg'].get('use_metric_loss', True))
    gradient_accumulation_steps = max(1, int(cfg['training_cfg'].get('gradient_accumulation_steps', 1)))
    for epoch in range(max(0, last_epoch), cfg['training_cfg']['training_epochs']):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            if i % gradient_accumulation_steps == 0:
                if len(train_loader) - i < gradient_accumulation_steps:
                    break
                optim_g.zero_grad()
                if use_metric_loss:
                    optim_d.zero_grad()
            if rank == 0 and i % gradient_accumulation_steps == 0:
                start_b = time.time()
            is_optimizer_step = (i + 1) % gradient_accumulation_steps == 0
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
            noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
            noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))
            one_labels = torch.ones(batch_size).to(device, non_blocking=True)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            # Discriminator
            # ------------------------------------------------------- #
            if use_metric_loss:
                audio_list_r = list(clean_audio.cpu().numpy())
                audio_list_g = list(audio_g.detach().cpu().numpy())
                batch_pesq_score, batch_pesq_mask = batch_pesq(audio_list_r, audio_list_g, cfg)
                metric_r = discriminator(clean_mag, clean_mag)
                metric_g = discriminator(clean_mag, mag_g.detach())
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
                metric_pesq_mask = batch_pesq_mask.to(device) if batch_pesq_mask is not None else None
                metric_pesq_valid_count = int(metric_pesq_mask.sum().item()) if metric_pesq_mask is not None else 0
                metric_pesq_invalid_count = clean_audio.size(0) - metric_pesq_valid_count
                if batch_pesq_score is not None:
                    loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten()[metric_pesq_mask])
                else:
                    loss_disc_g = metric_g.sum() * 0.0
                loss_disc_all = loss_disc_r + loss_disc_g
                (loss_disc_all / gradient_accumulation_steps).backward()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
            else:
                metric_pesq_mask = None
                metric_pesq_valid_count = 0
                metric_pesq_invalid_count = 0
                loss_disc_all = torch.zeros((), device=device)
                grad_norm_d = 0.0
            # ------------------------------------------------------- #

            # Generator
            # ------------------------------------------------------- #
            # Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            # L2 Complex Loss
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # Time Loss
            loss_time = F.l1_loss(clean_audio, audio_g)
            # Metric Loss
            if use_metric_loss:
                metric_g = discriminator(clean_mag, mag_g)
                if metric_pesq_mask is not None and torch.any(metric_pesq_mask):
                    loss_metric = F.mse_loss(metric_g.flatten()[metric_pesq_mask], one_labels[metric_pesq_mask])
                else:
                    loss_metric = metric_g.sum() * 0.0
            else:
                loss_metric = torch.zeros((), device=device)
            # Consistancy Loss
            _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True)
            loss_con = F.mse_loss(com_g, rec_com) * 2

            loss_gen_all = (
                loss_metric * cfg['training_cfg']['loss']['metric'] +
                loss_mag * cfg['training_cfg']['loss']['magnitude'] +
                loss_pha * cfg['training_cfg']['loss']['phase'] +
                loss_com * cfg['training_cfg']['loss']['complex'] +
                loss_time * cfg['training_cfg']['loss']['time'] +
                loss_con * cfg['training_cfg']['loss']['consistancy']
            )

            (loss_gen_all / gradient_accumulation_steps).backward()
            if use_metric_loss:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            if is_optimizer_step:
                grad_norm_g = grad_norm(generator.parameters())
                optim_g.step()
                if use_metric_loss:
                    grad_norm_d = grad_norm(discriminator.parameters())
                    optim_d.step()
            # ------------------------------------------------------- #

            if rank == 0 and is_optimizer_step:
                with torch.no_grad():
                    train_metrics = {
                        "loss": loss_gen_all,
                        "loss_generator": loss_gen_all,
                        "loss_discriminator": loss_disc_all,
                        "loss_metric": loss_metric,
                        "loss_magnitude": loss_mag,
                        "loss_phase": loss_pha,
                        "loss_complex": F.mse_loss(clean_com, com_g),
                        "loss_time": loss_time,
                        "loss_consistency": F.mse_loss(com_g, rec_com),
                        "metric_pesq_valid_count": metric_pesq_valid_count,
                        "metric_pesq_invalid_count": metric_pesq_invalid_count,
                        "grad_norm_g": grad_norm_g,
                        "grad_norm_d": grad_norm_d,
                    }
                    add_metric_window(train_metric_sums, train_metrics)
                    train_metric_count += 1
                    train_window_elapsed_sec += time.time() - start_b
                    samples_seen += clean_audio.size(0) * num_gpus * gradient_accumulation_steps
                    train_window_samples += clean_audio.size(0) * num_gpus * gradient_accumulation_steps
                    global_step = steps + 1
                # STDOUT logging
                if global_step % cfg['env_setting']['stdout_interval'] == 0:
                    print(
                        'Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric Loss: {:4.3f}, '
                        'Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, Cons Loss: {:4.3f}, s/b : {:4.3f}'.format(
                            global_step,
                            scalar_value(train_metrics["loss_generator"]),
                            scalar_value(train_metrics["loss_discriminator"]),
                            scalar_value(train_metrics["loss_metric"]),
                            scalar_value(train_metrics["loss_magnitude"]),
                            scalar_value(train_metrics["loss_phase"]),
                            scalar_value(train_metrics["loss_complex"]),
                            scalar_value(train_metrics["loss_time"]),
                            scalar_value(train_metrics["loss_consistency"]),
                            time.time() - start_b,
                        )
                    )

                # Checkpointing
                if global_step % checkpoint_interval == 0:
                    generator_state = {
                        'generator': (generator.module if num_gpus > 1 else generator).state_dict()
                    }
                    optimizer_state = {
                        'discriminator': (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'steps': steps,
                        'epoch': epoch,
                        'best_avqi_gap_to_clean': getattr(
                            generator.module if num_gpus > 1 else generator,
                            "_best_avqi_gap_to_clean",
                            float("inf"),
                        ),
                        'latest_avqi_gap_to_clean': getattr(
                            generator.module if num_gpus > 1 else generator,
                            "_latest_avqi_gap_to_clean",
                            None,
                        ),
                        'best_guarded_val_loss': getattr(
                            generator.module if num_gpus > 1 else generator,
                            "_best_guarded_val_loss",
                            float("inf"),
                        ),
                    }
                    exp_name = f"{args.exp_path}/g_{global_step:08d}.pth"
                    save_checkpoint(exp_name, generator_state)
                    exp_name = f"{args.exp_path}/do_{global_step:08d}.pth"
                    save_checkpoint(exp_name, optimizer_state)
                    prune_step_checkpoints(
                        args.exp_path,
                        keep=cfg['env_setting'].get('checkpoint_keep', 3),
                        prefixes=("g_", "do_"),
                    )

                if rank == 0 and global_step % avqi_interval == 0:
                    validation_generator = generator.module if num_gpus > 1 else generator
                    avqi_metrics = log_validation_avqi_metrics(validation_generator, cfg, device, global_step)
                    gap = avqi_metrics["avqi_gap_to_clean"]
                    (generator.module if num_gpus > 1 else generator)._latest_avqi_gap_to_clean = gap
                    print(
                        "Steps : {:d}, AVQI gap to clean: {:4.3f}".format(
                            global_step,
                            gap,
                        )
                    )
                    if wandb_run is not None:
                        avqi_log = {"charts/global_step": global_step, **format_wandb_avqi_metrics(avqi_metrics)}
                        wandb.log(avqi_log, step=global_step)
                    save_best_avqi_gap_checkpoints(
                        generator, discriminator, [optim_g, optim_d], args.exp_path, epoch, global_step, num_gpus, gap
                    )

                # Tensorboard summary logging
                if global_step % summary_interval == 0:
                    for name, value in train_metrics.items():
                        sw.add_scalar(f"Training/{name}", scalar_value(value), global_step)
                    if wandb_run is not None:
                        log_metrics = {
                            "charts/epoch": epoch + 1,
                            "charts/global_step": global_step,
                            "charts/lr": optim_g.param_groups[0]['lr'],
                            "charts/samples_per_sec": train_window_samples / train_window_elapsed_sec,
                        }
                        averaged_train_metrics = average_metric_window(train_metric_sums, train_metric_count)
                        if "grad_norm_g" in averaged_train_metrics:
                            log_metrics["charts/grad_norm"] = averaged_train_metrics["grad_norm_g"]
                        for name, value in averaged_train_metrics.items():
                            if name.startswith("metric_pesq"):
                                continue
                            if name.startswith("grad_norm"):
                                log_metrics[f"charts/{name}"] = value
                            else:
                                log_metrics[f"train/{name}"] = value
                        wandb.log(log_metrics, step=global_step)
                    train_metric_sums = {}
                    train_metric_count = 0
                    train_window_elapsed_sec = 0.0
                    train_window_samples = 0

                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

                # Validation
                if global_step % validation_interval == 0:
                    validation_generator = generator.module if num_gpus > 1 else generator
                    validation_generator.eval()
                    torch.cuda.empty_cache()
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            if len(batch) == 7:
                                clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, _ = batch
                            else:
                                clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
                            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
                            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))

                            mag_g, pha_g, com_g = validation_generator(noisy_mag.to(device), noisy_pha.to(device))

                            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g, cfg)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()

                        val_mag_err = val_mag_err_tot / (j+1)
                        val_pha_err = val_pha_err_tot / (j+1)
                        val_com_err = val_com_err_tot / (j+1)
                        loss_cfg = cfg['training_cfg']['loss']
                        val_loss = (
                            val_mag_err * loss_cfg.get('magnitude', 0.0)
                            + val_pha_err * loss_cfg.get('phase', 0.0)
                            + val_com_err * loss_cfg.get('complex', 0.0)
                        )
                        print('Steps : {:d}, Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, s/b : {:4.3f}'.
                                format(global_step, val_mag_err, val_pha_err, val_com_err, time.time() - start_b))
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, global_step)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, global_step)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, global_step)
                        if wandb_run is not None:
                            wandb.log({
                                "charts/epoch": epoch + 1,
                                "charts/global_step": global_step,
                                "val/loss": val_loss,
                                "val/loss_magnitude": val_mag_err,
                                "val/loss_phase": val_pha_err,
                                "val/loss_complex": val_com_err,
                            }, step=global_step)
                        save_best_guarded_checkpoints(
                            generator,
                            discriminator,
                            [optim_g, optim_d],
                            args.exp_path,
                            epoch,
                            global_step,
                            num_gpus,
                            val_loss,
                        )

                    validation_generator.train()

                    print(f"valid: Mag_loss {val_mag_err}, Phase_loss {val_pha_err}, Complex_loss {val_com_err}")

            if is_optimizer_step:
                steps += 1

        scheduler_g.step()
        if use_metric_loss:
            scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='SEMamba_advanced')
    parser.add_argument('--config', default='recipes/SEMamba_advanced/SEMamba_advanced.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        warnings.warn(
            f"Warning: The actual number of available GPUs ({available_gpus}) is less than the .yaml config ({num_gpus}). Auto reset to num_gpu = {available_gpus}",
            UserWarning
        )
        cfg['env_setting']['num_gpus'] = available_gpus
        num_gpus = available_gpus
        time.sleep(5)


    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)

    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_available_gpus}")
        print_gpu_info(num_available_gpus, cfg)
    else:
        print("CUDA is not available.")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(0, args, cfg)

if __name__ == '__main__':
    main()
