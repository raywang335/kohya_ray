export IMG_DIR='/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/df_data/ray_tools/server_filtered_images_after_dect_opencv.txt'
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/deepfashion/df_all.json"
export TEXT_DIR="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/deepfashion/texts"
python merge_captions_to_metadata_df.py $IMG_DIR $META_FILE --caption_extention='.txt' --full_path --text_dir=$TEXT_DIR