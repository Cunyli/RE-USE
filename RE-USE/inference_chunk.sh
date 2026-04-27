CUDA_VISIBLE_DEVICES='0' python ./inference_chunk.py \
   --input_folder ./long_noisy_audio \
   --output_folder ./long_enhanced_audio \
   --checkpoint_file ./exp/30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_peak_GAN_tel_mic/g_01134000.pth  \
   --config ./recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml \
   --chunk_size_in_seconds 5\
   --hop_length_portion 0.5\
   #--BWE 32000 \
   