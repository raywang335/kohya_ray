# %%

import numpy as np
import torch
from tqdm import tqdm
from library import model_util, sdxl_model_util

adapter_ckpt_path = "/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/sdxladapter_laion_12m_1024_v0/at-step00000200.ckpt"
sd_adapter, _ = sdxl_model_util.load_adapters_from_sdxl_checkpoint(sdxl_model_util.MODEL_VERSION_SDXL_BASE_V0_9, adapter_ckpt_path, "cpu")


# %%
from PIL import Image
img_path = '/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/lineart_demo_set/img/001144293.jpg'
x = torch.FloatTensor(np.array(Image.open(img_path).convert('L').resize((1024, 1024)))).unsqueeze(0).unsqueeze(0)
print(x.shape)
y = sd_adapter(x)
# %%
for name, param in sd_adapter.named_parameters():
    print(name, param.shape)
# %%
for feature in y:
    print(f'{feature.shape} {feature.dtype} {feature.device}')
# %%
# plt the features
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plot_feature(feature, title):
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(feature, cmap='gray')
    ax.set_title(title)
    plt.show()
features = [i.squeeze(0).detach().numpy() for i in y]
for i in range(features[0].shape[0]):
    # plot_feature(np.mean(features[0][i]), "feature")
    print(np.mean(features[0][i]), "feature")
# %%
