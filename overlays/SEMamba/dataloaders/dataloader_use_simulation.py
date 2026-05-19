import sys
from pathlib import Path

import torch

from models.stfts import mag_phase_stft
from models.pcs400 import cal_pcs


def _add_use_simulation_to_path(root):
    root = Path(root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class USESimulationPairDataset(torch.utils.data.Dataset):
    """SEMamba training adapter for fixed pairs exported by USE_simulation."""

    def __init__(
        self,
        pair_manifest,
        use_simulation_root="../USE_simulation",
        sampling_rate=16000,
        segment_size=32000,
        n_fft=400,
        hop_size=100,
        win_size=400,
        compress_factor=1.0,
        split=True,
        random_start=True,
        normalize=True,
        pcs=False,
        seed=1234,
        mode="train",
        return_metadata=False,
    ):
        _add_use_simulation_to_path(use_simulation_root)
        from use_simulation_datasets import FixedPairDataset

        self.sampling_rate = int(sampling_rate)
        self.segment_size = int(segment_size)
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_size)
        self.win_size = int(win_size)
        self.compress_factor = float(compress_factor)
        self.split = bool(split)
        self.random_start = bool(random_start)
        self.pcs = bool(pcs)
        self.return_metadata = bool(return_metadata)
        self.dataset = FixedPairDataset(
            pair_manifest=pair_manifest,
            wav_len=None,
            num_per_epoch=0,
            random_start=False,
            target_sample_rate=self.sampling_rate,
            mode=mode,
            normalize=normalize,
            seed=seed,
        )

    def __getitem__(self, index):
        noisy_audio, clean_audio, metadata = self.dataset[index]
        noisy_audio = torch.as_tensor(noisy_audio, dtype=torch.float32).reshape(1, -1)
        clean_audio = torch.as_tensor(clean_audio, dtype=torch.float32).reshape(1, -1)

        if self.pcs:
            clean_audio = torch.as_tensor(cal_pcs(clean_audio.squeeze().numpy()), dtype=torch.float32).reshape(1, -1)

        norm_factor = torch.sqrt(noisy_audio.numel() / torch.sum(noisy_audio ** 2.0).clamp_min(1e-12))
        clean_audio = clean_audio * norm_factor
        noisy_audio = noisy_audio * norm_factor

        clean_audio, noisy_audio = self._crop_or_pad_pair(clean_audio, noisy_audio)

        clean_mag, clean_pha, clean_com = mag_phase_stft(
            clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )
        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )

        item = (
            clean_audio.squeeze(),
            clean_mag.squeeze(),
            clean_pha.squeeze(),
            clean_com.squeeze(),
            noisy_mag.squeeze(),
            noisy_pha.squeeze(),
        )
        if self.return_metadata:
            return item + (metadata,)
        return item

    def __len__(self):
        return len(self.dataset)

    def _crop_or_pad_pair(self, clean_audio, noisy_audio):
        length = min(clean_audio.size(1), noisy_audio.size(1))
        clean_audio = clean_audio[:, :length]
        noisy_audio = noisy_audio[:, :length]

        if not self.split:
            return clean_audio, noisy_audio

        if length >= self.segment_size:
            if self.random_start:
                start = int(torch.randint(0, length - self.segment_size + 1, (1,)).item())
            else:
                start = 0
            return (
                clean_audio[:, start : start + self.segment_size],
                noisy_audio[:, start : start + self.segment_size],
            )

        pad = self.segment_size - length
        clean_audio = torch.nn.functional.pad(clean_audio, (0, pad), "constant")
        noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad), "constant")
        return clean_audio, noisy_audio
