export IMG_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/deepfashion/cache"
export MODEL_DIR="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/deepfashion/df_all.json"
export OUT_META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/deepfashion/df_all_cache.json"
export IMAGE_LIST="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/df_data/ray_tools/cache_filtered_images_after_dect_opencv.txt"
export TEXT_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/deepfashion/texts"

CUDA_VISIBLE_DEVICES=3 python prepare_text_data_df.py   $IMG_DIR \
                                    $META_FILE \
                                    $OUT_META_FILE \
                                    $MODEL_DIR \
                                    --batch_size 5 \
                                    --image_list=$IMAGE_LIST \
                                    --text_dir=$TEXT_DIR \
                                    --skip_existing