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
import math
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
        for fname in get_filepaths(args.input_folder):
            print(fname)
            try:
                os.makedirs(args.output_folder + fname[0:fname.rfind('/')].replace(args.input_folder,''), exist_ok=True)
                Noisy_wav, noisy_sr = torchaudio.load(fname)
            except Exception as e:
                print(f"Warning: cannot read {fname}, skipping. ({e})")
                continue

            if args.BWE is not None:
                opts = {"res_type": "kaiser_best"}
                Noisy_wav = librosa.resample(Noisy_wav.cpu().numpy(), orig_sr=noisy_sr, target_sr=int(args.BWE), **opts)
                noisy_sr = int(args.BWE)
            
            chunk_size = int(args.chunk_size_in_seconds*noisy_sr) # (in samples)
            hop_length = int(args.hop_length_portion*chunk_size) # (in samples)
            window = torch.hann_window(chunk_size).to(device)

            n_fft_scaled = make_even(n_fft * noisy_sr // sampling_rate)
            hop_size_scaled = make_even(hop_size * noisy_sr // sampling_rate)
            win_size_scaled = make_even(win_size * noisy_sr // sampling_rate)

            Noisy_wav = torch.FloatTensor(Noisy_wav).to(device)
            audio_enhanced = torch.zeros_like(Noisy_wav).to(device)
            #norm = torch.zeros_like(Noisy_wav).to(device)
            window_sum = torch.zeros_like(Noisy_wav).to(device)
            for c in range(Noisy_wav.shape[0]): # for multi-channel speech
                noisy_wav = Noisy_wav[c:c+1,:]
                for i in range(max(1, math.ceil((noisy_wav.shape[1]-chunk_size)/hop_length)+1)):
                    noisy_wav_chunk = noisy_wav[:, i*hop_length : i*hop_length+chunk_size]

                    noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
                        noisy_wav_chunk,
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
                    audio_g = pad_or_trim_to_match(noisy_wav_chunk.detach(), audio_g, pad_value=1e-8)  # Align lengths using epsilon padding

                    audio_enhanced[c:c+1,i*hop_length:i*hop_length+chunk_size] += audio_g*window[0:audio_g.shape[1]] 
                    window_sum[c:c+1,i*hop_length:i*hop_length+chunk_size] += window[0:audio_g.shape[1]]
                    #norm[c:c+1,i*hop_length:i*hop_length+chunk_size] += 1.0
            nonzero_indices = (window_sum > 1e-8)
            audio_enhanced[:,nonzero_indices[0]] = audio_enhanced[:,nonzero_indices[0]]/window_sum[:,nonzero_indices[0]]
            assert audio_enhanced.shape == Noisy_wav.shape, audio_enhanced.shape
            output_file = os.path.join(args.output_folder + fname.replace(args.input_folder,'').split('.')[0]+'.flac') # save to .flac format
            torchaudio.save(output_file, audio_enhanced.cpu(), noisy_sr)

def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder')
    parser.add_argument('--output_folder')
    parser.add_argument('--config')
    parser.add_argument('--checkpoint_file')
    parser.add_argument('--chunk_size_in_seconds', type=float)
    parser.add_argument('--hop_length_portion', type=float)
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

