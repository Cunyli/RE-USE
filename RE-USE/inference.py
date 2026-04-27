# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import argparse
import torch
import torchaudio
import torch.nn as nn
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config, pad_or_trim_to_match
from huggingface_hub import hf_hub_download
RELU = nn.ReLU()

config_path = hf_hub_download(repo_id="nvidia/RE-USE", filename="config.json")

def get_filepaths(directory, file_type=None):
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            if file_type is not None:
                if filepath.split('.')[-1] == file_type:
                    file_paths.append(filepath)  # Add it to the list.
            else:
                file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.

def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1

def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    SE_model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=cfg).to(device)
    SE_model.eval()

    os.makedirs(args.output_folder, exist_ok=True)
    with torch.no_grad():
        for i, fname in enumerate(get_filepaths(args.input_folder)):
            print(fname)
            try:
                os.makedirs(args.output_folder + fname[0:fname.rfind('/')].replace(args.input_folder,''), exist_ok=True)
                noisy_wav, noisy_sr = torchaudio.load(fname)
            except Exception as e:
                print(f"Warning: cannot read {fname}, skipping. ({e})")
                continue
            
            if args.BWE is not None:
                opts = {"res_type": "kaiser_best"}
                noisy_wav = librosa.resample(noisy_wav.cpu().numpy(), orig_sr=noisy_sr, target_sr=int(args.BWE), **opts)
                noisy_sr = int(args.BWE)

            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            n_fft_scaled = make_even(n_fft * noisy_sr // sampling_rate)
            hop_size_scaled = make_even(hop_size * noisy_sr // sampling_rate)
            win_size_scaled = make_even(win_size * noisy_sr // sampling_rate)

            noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
                noisy_wav,
                n_fft=n_fft_scaled,
                hop_size=hop_size_scaled,
                win_size=win_size_scaled,
                compress_factor=compress_factor,
                center=True,
                addeps=False
            )
            amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha)
            # To remove "strange sweep artifact"
            mag = torch.expm1(RELU(amp_g)) # [1, F, T]
            zero_portion = torch.sum(mag==0, 1)/mag.shape[1]
            amp_g[:,:,(zero_portion>0.5)[0]] = 0

            audio_g = mag_phase_istft(amp_g, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, compress_factor)
            audio_g = pad_or_trim_to_match(noisy_wav.detach(), audio_g, pad_value=1e-8)  # Align lengths using epsilon padding
            assert audio_g.shape == noisy_wav.shape, audio_g.shape

            output_file = os.path.join(args.output_folder + fname.replace(args.input_folder,'').split('.')[0]+'.wav') # save to .flac format
            torchaudio.save(output_file, audio_g.cpu(), noisy_sr)
            
def main():
    print('Initializing Inference Process...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder')
    parser.add_argument('--output_folder')
    parser.add_argument('--config')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--BWE', default=None)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("Currently, CPU mode is not supported.")
        
    inference(args, device)


if __name__ == '__main__':
    main()
