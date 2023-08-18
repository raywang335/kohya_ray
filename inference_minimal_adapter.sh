export pormpt="raining night ,village road"
export lineart_set="lineart_demo_set_2"
export output_dir="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/$lineart_set/samples"
# export model_path="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/finetuned_xl_base/at-step00010000.ckpt"
# export model_path="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/laion_12m_1024_v0/epoch-000004.ckpt"
export model_path="/mnt/nfs/file_server/public/liujia/Models/StableDiffusionXL/SDXL_1_0/sd_xl_base_1.0.safetensors"

# export model_path="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/deepfashion_v01/epoch-000003.ckpt"
#--prompt="raining night, village road, flat design" 
export input_file="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/$lineart_set/demo.txt"
export adapter_path="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/sdxladapter_laion_12m_1024_v1/at-step00015000.ckpt"
export lineart_dir="/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/$lineart_set/img"
export seed=42
CUDA_VISIBLE_DEVICES=2 python sdxl_minimal_inference_adapter.py --ckpt_path=$model_path \
                    --prompt="a photo of a old woman" \
                    --output_dir=$output_dir \
                    --negative_prompt="" \
                    --seed=$seed \
                    --batchsize=4 \
                    --gen_number=1 \
                    --adapter_ckpt_path=$adapter_path \
                    --lineart_path=$lineart_dir \
                    --input_file=$input_file 

# CUDA_VISIBLE_DEVICES=0 python sdxl_minimal_inference_fulladapter.py --ckpt_path=$model_path \
#                     --prompt="a photo of a old woman" \
#                     --output_dir=$output_dir \
#                     --negative_prompt="" \
#                     --seed=$seed \
#                     --batchsize=1 \
#                     --gen_number=1 \
#                     --batchsize 1 \
#                     --adapter_ckpt_path=$adapter_path \
#                     --lineart_path=$lineart_dir \
#                     --input_file=$input_file 

