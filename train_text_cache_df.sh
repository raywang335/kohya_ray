export IMG_DIR='/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir'
export MODEL_DIR="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/hires_cache_dir_cache.json"
export OUT_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/laion_12m_1024_v0"

accelerate launch sdxl_train_with_text_cache.py --pretrained_model_name_or_path=$MODEL_DIR \
                                                --in_json $META_FILE \
                                                --learning_rate=4e-7 --train_batch_size=5 \
                                                --diffusers_xformers --gradient_checkpointing \
                                                --optimizer_type="AdamW" \
                                                --save_every_n_epochs=1 \
                                                --mixed_precision=bf16 \
                                                --full_bf16 \
                                                --max_train_steps=20000 \
                                                --train_data_dir=$IMG_DIR \
                                                --output_dir=$OUT_DIR \
                                                --logging_dir=$OUT_DIR/log \
                                                --cache_latents \
                                                --text_encoder_cache \
                                                --log_with='all' \
                                                --caption_dropout_rate=0.1                                                