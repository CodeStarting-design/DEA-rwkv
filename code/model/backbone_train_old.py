import torch.nn as nn
import torch.nn.functional as F

from .modules import DEABlockTrain, DEBlockTrain, CGAFusion

from .vrwkv6 import Block as VRWKV


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DEANet(nn.Module):
    def __init__(self, base_dim=32):
        super(DEANet, self).__init__()
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1
        self.down_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)
        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)
        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_VRWKV1 = VRWKV(base_dim*4, 2, 8, 0)
        self.level3_block2 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_VRWKV2 = VRWKV(base_dim*4, 2, 8, 1)
        self.level3_block3 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_VRWKV3 = VRWKV(base_dim*4, 2, 8, 2)
        self.level3_block4 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_VRWKV4 = VRWKV(base_dim*4, 2, 8, 3)
        self.level3_block5 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_VRWKV5 = VRWKV(base_dim*4, 2, 8, 4)
        self.level3_block6 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_VRWKV6 = VRWKV(base_dim*4, 2, 8, 5)
        self.level3_block7 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_VRWKV7 = VRWKV(base_dim*4, 2, 8, 6)
        self.level3_block8 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_VRWKV8 = VRWKV(base_dim*4, 2, 8, 7)
        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self.mix1 = CGAFusion(base_dim * 4, reduction=8)
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)

    def forward(self, x):
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)

        B, C, H, W = x_down3.shape
        # 将 H 和 W 保存到 patch_resolution 中
        patch_resolution = (H, W)

        x1 = self.level3_block1(x_down3_init)
        x1 = x1.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x1 = self.level3_VRWKV1(x1, patch_resolution)
        x1 = x1.view(B, H, W, C).permute(0, 3, 1, 2)

        x2 = self.level3_block2(x1)
        x2 = x2.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x2 = self.level3_VRWKV2(x2, patch_resolution)
        x2 = x2.view(B, H, W, C).permute(0, 3, 1, 2)

        x3 = self.level3_block3(x2)
        x3 = x3.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x3 = self.level3_VRWKV3(x3, patch_resolution)
        x3 = x3.view(B, H, W, C).permute(0, 3, 1, 2)

        x4 = self.level3_block4(x3)
        x4 = x4.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x4 = self.level3_VRWKV4(x4, patch_resolution)
        x4 = x4.view(B, H, W, C).permute(0, 3, 1, 2)

        x5 = self.level3_block5(x4)
        x5 = x5.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x5 = self.level3_VRWKV5(x5, patch_resolution)
        x5 = x5.view(B, H, W, C).permute(0, 3, 1, 2)

        x6 = self.level3_block6(x5)
        x6 = x6.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x6 = self.level3_VRWKV6(x6, patch_resolution)
        x6 = x6.view(B, H, W, C).permute(0, 3, 1, 2)

        x7 = self.level3_block7(x6)
        x7 = x7.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x7 = self.level3_VRWKV7(x7, patch_resolution)
        x7 = x7.view(B, H, W, C).permute(0, 3, 1, 2)

        x8 = self.level3_block8(x7)
        x8 = x8.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x8 = self.level3_VRWKV8(x8, patch_resolution)
        x8 = x8.view(B, H, W, C).permute(0, 3, 1, 2)
        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out