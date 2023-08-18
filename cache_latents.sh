export IMG_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/demo/highres_imgs/cache"
export MODEL_DIR="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/demo/highres_imgs/demo.json"
export OUT_META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/demo/highres_imgs/demo_cache.json"

CUDA_VISIBLE_DEVICES=0 python prepare_text_data.py   $IMG_DIR \
                                    $META_FILE \
                                    $OUT_META_FILE \
                                    $MODEL_DIR \
                                    --batch_size 5 \
                                    --full_path \
                                    --max_resolution 1024,1024