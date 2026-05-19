import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import defaultdict, deque
import os
import time
import argparse
import json
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from joblib import Parallel, delayed
from pesq import pesq

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

try:
    import wandb
except ImportError:
    wandb = None

torch.backends.cudnn.benchmark = True

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

def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_cfg']['training_epochs'])
        batch_size = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
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
        drop_last=True if train else False
    )


class WindowAverager:
    def __init__(self, window):
        self.window = int(window)
        self.values = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, metrics):
        for key, value in metrics.items():
            self.values[key].append(float(value))

    def averages(self):
        return {
            key: sum(values) / len(values)
            for key, values in self.values.items()
            if values
        }


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


def pesq_score_with_details(utts_r, utts_g, cfg, metadata=None):
    sampling_rate = cfg['stft_cfg']['sampling_rate']
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
        delayed(eval_pesq)(i, utts_r[i], utts_g[i]) for i in range(len(utts_r))
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
    return mean_score, len(valid_scores), invalid


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

    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    if cfg['training_cfg'].get('use_pretrainedD', False):
        discriminator.load_state_dict( torch.load('ckpts/pretrained_discriminator.pth') )
        print("Loaded pretrained weight from ckpts/pretrained_discriminator.pth.")

    # Create optimizer and schedulers
    optimizers = setup_optimizers((generator, discriminator), cfg)
    load_optimizer_states(optimizers, state_dict_do)
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = setup_schedulers(optimizers, cfg, last_epoch)

    # Create trainset and train_loader
    trainset = create_dataset(cfg, train=True, split=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True)

    # Create validset and validation_loader if rank is 0
    if rank == 0:
        validset = create_dataset(cfg, train=False, split=False, device=device)
        validation_loader = create_dataloader(validset, cfg, train=False)
        sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))
        wandb_run = None
        wandb_cfg = cfg.get('wandb_cfg', {})
        if wandb_cfg.get('use_wandb', False):
            if wandb is None:
                raise RuntimeError("wandb_cfg.use_wandb=true but wandb is not installed.")
            wandb_init_kwargs = {
                "project": wandb_cfg.get('project', 'reuse'),
                "entity": wandb_cfg.get('entity'),
                "name": wandb_cfg.get('run_name', args.exp_name),
                "mode": wandb_cfg.get('mode', 'online'),
                "tags": wandb_cfg.get('tags', []),
                "config": cfg,
            }
            if wandb_cfg.get('run_id'):
                wandb_init_kwargs["id"] = wandb_cfg['run_id']
                wandb_init_kwargs["resume"] = wandb_cfg.get('resume', 'allow')
            wandb_run = wandb.init(**wandb_init_kwargs)
            wandb.define_metric("steps")
            wandb.define_metric("Training/*", step_metric="steps")
            wandb.define_metric(f"TrainingAvg{cfg['env_setting'].get('summary_avg_window', 100)}/*", step_metric="steps")
            wandb.define_metric("Validation/*", step_metric="steps")
    else:
        wandb_run = None

    generator.train()
    discriminator.train()

    best_pesq, best_pesq_step = 0.0, 0
    train_averager = WindowAverager(cfg['env_setting'].get('summary_avg_window', 100))
    for epoch in range(max(0, last_epoch), cfg['training_cfg']['training_epochs']):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
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
            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
            batch_pesq_score, batch_pesq_mask = batch_pesq(audio_list_r, audio_list_g, cfg)

            # Discriminator
            # ------------------------------------------------------- #
            optim_d.zero_grad()
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

            loss_disc_all.backward()
            optim_d.step()
            # ------------------------------------------------------- #

            # Generator
            # ------------------------------------------------------- #
            optim_g.zero_grad()

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
            metric_g = discriminator(clean_mag, mag_g)
            if metric_pesq_mask is not None and torch.any(metric_pesq_mask):
                loss_metric = F.mse_loss(metric_g.flatten()[metric_pesq_mask], one_labels[metric_pesq_mask])
            else:
                loss_metric = metric_g.sum() * 0.0
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

            loss_gen_all.backward()
            optim_g.step()
            # ------------------------------------------------------- #

            if rank == 0:
                with torch.no_grad():
                    train_metrics = {
                        "Generator Loss": loss_gen_all.item(),
                        "Discriminator Loss": loss_disc_all.item(),
                        "Metric Loss": loss_metric.item(),
                        "Magnitude Loss": loss_mag.item(),
                        "Phase Loss": loss_pha.item(),
                        "Complex Loss": F.mse_loss(clean_com, com_g).item(),
                        "Time Loss": loss_time.item(),
                        "Consistancy Loss": F.mse_loss(com_g, rec_com).item(),
                        "Metric PESQ Valid Count": metric_pesq_valid_count,
                        "Metric PESQ Invalid Count": metric_pesq_invalid_count,
                    }
                train_averager.update({
                    name: value
                    for name, value in train_metrics.items()
                    if not name.endswith(" Count")
                })

                # STDOUT logging
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    print(
                        'Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric Loss: {:4.3f}, '
                        'Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, Cons Loss: {:4.3f}, s/b : {:4.3f}'.format(
                            steps,
                            train_metrics["Generator Loss"],
                            train_metrics["Discriminator Loss"],
                            train_metrics["Metric Loss"],
                            train_metrics["Magnitude Loss"],
                            train_metrics["Phase Loss"],
                            train_metrics["Complex Loss"],
                            train_metrics["Time Loss"],
                            train_metrics["Consistancy Loss"],
                            time.time() - start_b,
                        )
                    )

                # Checkpointing
                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    generator_state = {
                        'generator': (generator.module if num_gpus > 1 else generator).state_dict()
                    }
                    optimizer_state = {
                        'discriminator': (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'steps': steps,
                        'epoch': epoch
                    }
                    exp_name = f"{args.exp_path}/g_{steps:08d}.pth"
                    save_checkpoint(exp_name, generator_state)
                    exp_name = f"{args.exp_path}/do_{steps:08d}.pth"
                    save_checkpoint(exp_name, optimizer_state)
                    prune_step_checkpoints(
                        args.exp_path,
                        keep=cfg['env_setting'].get('checkpoint_keep', 3),
                        prefixes=("g_", "do_"),
                    )

                # Tensorboard summary logging
                if steps % cfg['env_setting']['summary_interval'] == 0:
                    for name, value in train_metrics.items():
                        sw.add_scalar(f"Training/{name}", value, steps)
                    avg_metrics = train_averager.averages()
                    for name, value in avg_metrics.items():
                        sw.add_scalar(f"TrainingAvg{train_averager.window}/{name}", value, steps)
                    if wandb_run is not None:
                        log_metrics = {"steps": steps}
                        log_metrics.update({f"Training/{name}": value for name, value in train_metrics.items()})
                        log_metrics.update({f"TrainingAvg{train_averager.window}/{name}": value for name, value in avg_metrics.items()})
                        wandb.log(log_metrics)

                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

                # Validation
                if steps % cfg['env_setting']['validation_interval'] == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    with torch.no_grad():
                        metadata_items = []
                        for j, batch in enumerate(validation_loader):
                            metadata = None
                            if len(batch) == 7:
                                clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha, metadata = batch
                            else:
                                clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
                            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
                            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))

                            mag_g, pha_g, com_g = generator(noisy_mag.to(device), noisy_pha.to(device))

                            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
                            audios_r += torch.split(clean_audio, 1, dim=0) # [1, T] * B
                            audios_g += torch.split(audio_g, 1, dim=0)
                            if metadata is not None:
                                metadata_items.append(metadata)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g, cfg)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()

                        val_mag_err = val_mag_err_tot / (j+1)
                        val_pha_err = val_pha_err_tot / (j+1)
                        val_com_err = val_com_err_tot / (j+1)
                        val_pesq_score, val_pesq_valid, val_pesq_invalid = pesq_score_with_details(
                            audios_r, audios_g, cfg, metadata_items
                        )
                        log_invalid_pesq_samples(
                            args.exp_path,
                            steps,
                            val_pesq_invalid,
                            cfg['env_setting'].get('pesq_invalid_log_limit', 5),
                        )
                        print('Steps : {:d}, PESQ Score: {:4.3f}, s/b : {:4.3f}'.
                                format(steps, val_pesq_score, time.time() - start_b))
                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        if wandb_run is not None:
                            wandb.log({
                                "steps": steps,
                                "Validation/PESQ Score": val_pesq_score,
                                "Validation/Magnitude Loss": val_mag_err,
                                "Validation/Phase Loss": val_pha_err,
                                "Validation/Complex Loss": val_com_err,
                            })

                    generator.train()

                    # Print best validation PESQ score in terminal
                    if val_pesq_score >= best_pesq:
                        best_pesq = val_pesq_score
                        best_pesq_step = steps
                        save_best_step_checkpoints(
                            args.exp_path,
                            steps,
                            {
                                'generator': (generator.module if num_gpus > 1 else generator).state_dict()
                            },
                            {
                                'discriminator': (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
                                'optim_g': optim_g.state_dict(),
                                'optim_d': optim_d.state_dict(),
                                'steps': steps,
                                'epoch': epoch,
                                'best_pesq': best_pesq,
                            },
                        )
                    print(f"valid: PESQ {val_pesq_score}, Mag_loss {val_mag_err}, Phase_loss {val_pha_err}. Best_PESQ: {best_pesq} at step {best_pesq_step}")

            steps += 1

        scheduler_g.step()
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
