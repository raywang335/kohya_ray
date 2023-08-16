from torch import nn
import torch


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x

class SdxlT2IAdapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280], nums_rb=2, cin=64, ksize=3, sk=False, use_conv=True):
        super(SdxlT2IAdapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        is_down = [True, False, False, False, True, False]
        # before
        # is_down = [False, False, True, False, True, False]
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=is_down[i*nums_rb+j], ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=is_down[i*nums_rb+j], ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x, inference=False):
        # unshuffle
        x = self.unshuffle(x)
        if inference:
            x = x.repeat(2,1,1,1)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)

        return features

class SdxlT2IAdapterFull(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=2, cin=64, ksize=3, sk=False, use_conv=True):
        super(SdxlT2IAdapterFull, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i == len(channels)-1) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
                elif (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x, inference=False):
        # unshuffle
        x = self.unshuffle(x)
        if inference:
            x = x.repeat(2,1,1,1)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)

        return features

# if __name__ == "__main__":
#     import time
#     IN_CHANNELS: int = 4
#     OUT_CHANNELS: int = 4
#     ADM_IN_CHANNELS: int = 2816
#     CONTEXT_DIM: int = 2048
#     MODEL_CHANNELS: int = 320
#     TIME_EMBED_DIM = 320 * 4

#     device = 'cuda:4'
#     print("create unet")
#     unet = SdxlUNet2DConditionModel()

#     unet.to(device)
#     unet.set_use_memory_efficient_attention(True, False)
#     unet.set_gradient_checkpointing(True)
#     unet.eval()

#     sdxl_adapter = SdxlT2IAdapter()
#     sdxl_adapter.to(device)
#     sdxl_adapter.train()

#     unet.to(dtype=torch.float16)
#     sdxl_adapter.to(dtype=torch.float16)
#     # 使用メモリ量確認用の疑似学習ループ
#     print("preparing optimizer")

#     # optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, nesterov=True, momentum=0.9) # not working

#     # import bitsandbytes
#     # optimizer = bitsandbytes.adam.Adam8bit(unet.parameters(), lr=1e-3)        # not working
#     # optimizer = bitsandbytes.optim.RMSprop8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2
#     # optimizer= bitsandbytes.optim.Adagrad8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2

#     import transformers

#     # optimizer = transformers.optimization.Adafactor(unet.parameters(), relative_step=True)  # working at 22.2GB with torch2
#     optimizer = transformers.optimization.AdamW(sdxl_adapter.parameters())  # working at 41.7GB with torch2

#     scaler = torch.cuda.amp.GradScaler(enabled=True)

#     print("start training")
#     steps = 10
#     batch_size = 2

#     for step in range(steps):
#         print(f"step {step}")
#         if step == 1:
#             time_start = time.perf_counter()

#         x = torch.randn(batch_size, 4, 128, 128).to(device)  # 1024x1024
#         t = torch.randint(low=0, high=10, size=(batch_size,), device=device)
#         ctx = torch.randn(batch_size, 77, 2048).to(device)
#         y = torch.randn(batch_size, ADM_IN_CHANNELS).to(device)

#         lineart_img = torch.randn(batch_size, 1, 1024, 1024).to(device)

#         with torch.cuda.amp.autocast(enabled=True):
#             ada_cond = sdxl_adapter(lineart_img)
#             output = unet(x, t, ctx, y, adapter_features=ada_cond)
#             target = torch.randn_like(output)
#             loss = torch.nn.functional.mse_loss(output, target)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)

#     time_end = time.perf_counter()
#     print(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")
