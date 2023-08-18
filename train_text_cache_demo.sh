export IMG_DIR='/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/demo/highres_imgs/data'
export MODEL_DIR="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/demo/highres_imgs/output_cache.json"
export OUT_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/demo/highres_imgs/model/v0"
# export RESUME_PATH="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/fine_tuned_xl_base_3m/at-step00070000.ckpt"
# export RESUME_PATH="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/fine_tuned_xl_base_3m/at-step00070000.ckpt"

OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=2 accelerate launch sdxl_train_with_text_cache.py --pretrained_model_name_or_path=$MODEL_DIR \
                                                --in_json $META_FILE \
                                                --optimizer_type="Adafactor" --optimizer_args="scale_parameter=False relative_step=False warmup_init=False" \
                                                --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 --learning_rate=1e-7 --train_batch_size=1 \
                                                --diffusers_xformers --gradient_checkpointing\
                                                --save_every_n_steps=10000 \
                                                --mixed_precision=bf16 \
                                                --full_bf16 \
                                                --max_train_steps=300000 \
                                                --train_data_dir=$IMG_DIR \
                                                --output_dir=$OUT_DIR \
                                                --logging_dir=$OUT_DIR/log \
                                                --cache_latents \
                                                --text_encoder_cache \
                                                