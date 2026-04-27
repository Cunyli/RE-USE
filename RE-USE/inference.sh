CUDA_VISIBLE_DEVICES='0' python ./inference.py \
   --input_folder ./noisy_audio \
   --output_folder ./enhanced_audio \
   --checkpoint_file ./exp/30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_peak_GAN_tel_mic/g_01134000.pth  \
   --config ./recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml \
   #--BWE 32000 \
