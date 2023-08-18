export IMG_DIR='/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/cache_dir'
export MODEL_DIR="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/finetuen_laion_3m/kohya_cache.json"
export OUT_DIR="./fine_tuned_xl_base_3m_1e-5_ds"
OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=1,2 accelerate launch --config_file=./config_files/accelerate/default_config.yaml sdxl_train_with_text_cache.py --pretrained_model_name_or_path=$MODEL_DIR \
                                                --in_json $META_FILE \
                                                --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 --learning_rate=1e-5 \
                                                --diffusers_xformers --gradient_checkpointing\
                                                --save_every_n_steps=10000 \
                                                --mixed_precision=fp16 \
                                                --max_train_steps=300000 \
                                                --train_data_dir=$IMG_DIR \
                                                --output_dir=$OUT_DIR \
                                                --logging_dir=$OUT_DIR/log \
                                                --cache_latents \
                                                --text_encoder_cache \
                                                --train_batch_size=2
                                                