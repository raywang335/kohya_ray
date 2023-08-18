export IMG_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/finetuen_laion_3m"
export MODEL_DIR="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/hires_cache_dir.json"
export OUT_META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/hires_cache_dir_cache.json"
export CACHE_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/cache"
CUDA_VISIBLE_DEVICES=6 python prepare_text_data_df.py   $IMG_DIR \
                                    $META_FILE \
                                    $OUT_META_FILE \
                                    $MODEL_DIR \
                                    --max_resolution 1024,1024 \
                                    --batch_size 2 \
                                    --skip_existing \
                                    --max_bucket_reso 1280 \
                                    --min_bucket_reso 512 \
                                    --cache_dir $CACHE_DIR \
                                    --full_path \
                                    --null_token