export pormpt="a rabbit working in a factory."
export output_dir="./samples_xl_finetune"
# export model_path="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/finetuned_xl_base/at-step00010000.ckpt"
export model_path="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"

export seed=41231
python sdxl_gen_img.py  --from_file="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/prompts_file.txt"  \
                        --outdir=$output_dir \
                        --H=1024 --W=1024 --steps=30 --batch_size=4 \
                        --sampler='euler_a' \
                        --scale=7.5 \
                        --ckpt=$model_path \
                        --seed=$seed \
                        --fp16 \
                        --xformers \
                        --original_height=1024 --original_width=1024 \
                        --crop_top=0 --crop_left=0 \
                        --no_half_vae

