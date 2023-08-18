export pormpt="raining night ,village road"
export output_dir="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/laion_12m_1024_v0/samples"
# export model_path="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/finetuned_xl_base/at-step00010000.ckpt"
export model_path="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/laion_12m_1024_v0/epoch-000004.ckpt"
# export model_path="/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"

# export model_path="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/deepfashion_v01/epoch-000003.ckpt"
#--prompt="raining night, village road, flat design" 
export input_file="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/prompts_file.txt"

export seed=41231
CUDA_VISIBLE_DEVICES=3 python sdxl_minimal_inference.py --ckpt_path=$model_path \
                    --prompt="winter morning, food factory, flat design, rabbit worker" \
                    --output_dir=$output_dir \
                    --negative_prompt="" \
                    --seed=$seed \
                    --batchsize=1 \
                    --gen_number=4 \
                    --input_file=$input_file \
                    --batchsize 4

