export IMG_DIR='/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir'
export MODEL_DIR="/mnt/nfs/file_server/public/liujia/Models/StableDiffusionXL/SDXL_1_0/sd_xl_base_1.0.safetensors"
# export MODEL_DIR="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/hires_cache_demo_cache.json"
export OUT_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/sdxladapter_laion_12m_1024_v3"
export lineart_dir="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/T2Iadapter_lineart/lineart_data"

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 29590 sdxl_train_adapter_with_text_cache.py --pretrained_model_name_or_path=$MODEL_DIR \
                                                --in_json $META_FILE \
                                                --learning_rate=1e-4 --train_batch_size=5 \
                                                --lr_scheduler="polynomial" \
                                                --lr_warmup_steps=500 \
                                                --lr_scheduler_power=1 \
                                                --diffusers_xformers --gradient_checkpointing \
                                                --optimizer_type="AdamW" \
                                                --save_every_n_steps=500 \
                                                --mixed_precision=bf16 \
                                                --max_train_steps=8000 \
                                                --full_bf16 \
                                                --train_data_dir=$IMG_DIR \
                                                --output_dir=$OUT_DIR \
                                                --logging_dir=$OUT_DIR/log \
                                                --log_with='all' \
                                                --cache_latents \
                                                --text_encoder_cache \
                                                --caption_dropout_rate=0.1 \
                                                --lineart_dir=$lineart_dir  \
                                                --log_image_every_n_steps=25

                                                # --adapter_resume_path=None 