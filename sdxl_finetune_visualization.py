# %%
# 手元で推論を行うための最低限のコード。HuggingFace／DiffusersのCLIP、schedulerとVAEを使う
# Minimal code for performing inference at local. Use HuggingFace/Diffusers CLIP, scheduler and VAE

import argparse
import datetime
import math
import os
import random
from einops import repeat
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler
from PIL import Image
import open_clip
from safetensors.torch import load_file

from library import model_util, sdxl_model_util
import networks.lora as lora

# scheduler: このあたりの設定はSD1/2と同じでいいらしい
# scheduler: The settings around here seem to be the same as SD1/2
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

target_height = 1024
target_width = 1024
original_height = target_height
original_width = target_width
crop_top = 0
crop_left = 0
ckpt_path = "/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/laion_12m_1024_v0/epoch-000004.ckpt"
steps = 50
guidance_scale = 7


DEVICE = "cuda"
DTYPE = torch.float16  # bfloat16 may work

# HuggingFaceのmodel id
text_encoder_1_name = "openai/clip-vit-large-patch14"
text_encoder_2_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
seed = 1
# checkpointを読み込む。モデル変換についてはそちらの関数を参照
# Load checkpoint. For model conversion, see this function

# 本体RAMが少ない場合はGPUにロードするといいかも
# If the main RAM is small, it may be better to load it on the GPU
text_model1, text_model2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
    sdxl_model_util.MODEL_VERSION_SDXL_BASE_V0_9, ckpt_path, "cpu"
)
print(text_model1.state_dict().keys())
# %%
ori_ckpt_path = "/mnt/nfs/file_server/public/sdxl_0_9/hugging_face_file/SDXL_0_9/stable-diffusion-xl-base-0.9/sd_xl_base_0.9.safetensors"
text_model1, text_model2, vae, unet_ori, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
    sdxl_model_util.MODEL_VERSION_SDXL_BASE_V0_9, ori_ckpt_path, "cpu"
)
# %%
delta_dict = {}
int_len = 5
for key in unet_ori.state_dict().keys():
    split_res = key.split('.')
    if len(split_res) < int_len:
        continue
    # group1_key = split_res[0] + "." + key.split('.')[1]+ "." + key.split('.')[2]+ "." + key.split('.')[3]
    group1_key = '.'.join(split_res[:int_len])
    if group1_key in delta_dict.keys():
        delta_dict[group1_key] += torch.mean(torch.abs(unet_ori.state_dict()[key] - unet.state_dict()[key]))
    else:
        delta_dict[group1_key] = torch.mean(torch.abs(unet_ori.state_dict()[key] - unet.state_dict()[key]))
import pprint
pprint.pprint(delta_dict)
# %%
import matplotlib.pyplot as plt
import numpy as np
def to_float(x):
    x = [i.item() for i in x]
    return np.array(x)
def bulid_mapping(x):
    mapper = {}
    i=0
    for key in x:
    
        mapper[key] = i
        i+=1
    return mapper
name = sorted(delta_dict, key=lambda k: delta_dict[k].item(), reverse=True)
data = to_float(delta_dict.values())
mapper = bulid_mapping(delta_dict.keys())
# data =  {'input_blocks': 21483.9473,
#  'label_emb': 91.9481,
#  'middle_block': 10326.6758,
#  'out': 0.5790,
#  'output_blocks': 37395.4023,
#  'time_embed': 24.9831}
plt.bar(delta_dict.keys(), data)
pprint.pprint(name[:20], width=1)
# %%
