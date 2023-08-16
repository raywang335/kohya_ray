# From https://github.com/carolineec/informative-drawings
# MIT License

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from einops import rearrange


norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector:
    def __init__(self):
        self.model = self.load_model('sk_model.pth')
        self.model_coarse = self.load_model('sk_model2.pth')

    def load_model(self, name):
        modelpath = os.path.join("/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/lineart_models", name)
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
        model.eval()
        model = model.cuda()
        return model

    def __call__(self, input_image, coarse):
        model = self.model_coarse if coarse else self.model
        if isinstance(input_image, str):
            input_image = np.array(Image.open(input_image))
            if len(input_image.shape) < 3:
                input_image = np.expand_dims(input_image, axis=2).repeat(3, axis=2)
            elif input_image.shape[2] == 4:
                input_image = input_image[:, :, :3]
        assert input_image.ndim == 3
        image = input_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().cuda()
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            line = model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

            return line

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    model = LineartDetector()
    input_image = Image.open('/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/laion_12m_aesthetics/00000/000000003.jpg')
    input_image = np.array(input_image)
    line = model(input_image, coarse=False)
    Image.fromarray(line).save('/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/sd_xl_test/kohya_ss/test.png')