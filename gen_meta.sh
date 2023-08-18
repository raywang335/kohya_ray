export IMG_DIR='/mnt/nfs/file_server/public/lipengxiang/improved_aesthetics_6plus_out'
export META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/hires_cache_dir.json"
export IN_META_FILE="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/hires_cache_dir.json"
python merge_captions_to_metadata.py $IMG_DIR $META_FILE --caption_extention='.txt' --full_path 