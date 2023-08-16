# %%
from library import model_util, sdxl_model_util
adapter_ckpt_path = "/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/sdxladapter_laion_12m_1024_v0.3/at-step00020000.ckpt"
sd_adapter, _ = sdxl_model_util.load_adapters_from_sdxl_checkpoint(sdxl_model_util.MODEL_VERSION_SDXL_BASE_V0_9, adapter_ckpt_path, "cpu")
sd_adapter.cuda()
# %%
for key, value in sd_adapter.named_parameters():
    print(f'{key}: {value}')
    break
# %%
from PIL import Image
import numpy as np

img = Image.open("/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/test.png").convert('L')
print(img.size)

img = img.resize((img.size[1] // 4, img.size[0] // 4))
print(img.size)
img.save("/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/test_256.png")
# %%
import numpy as np
f = np.load('/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/hires_cache_dir/cache/000000003.npz', allow_pickle=True)['latents']
print(f.shape)
# %%
