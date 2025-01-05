import torch.nn as nn
import torch.nn.functional as F

from .modules import DEABlockTrain, DEBlockTrain, CGAFusion

from .nvrwkv6 import Block as RWKV,RLN,RLN1D


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
        self.level2_RWKV1 = RWKV(base_dim*2,2,2,0,RLN1D)
        self.level2_RWKV2 = RWKV(base_dim*2,2,2,1,RLN1D)
        self.up_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)
        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = DEABlockTrain(default_conv, base_dim * 4, 3)
        # self.level3_RWKV1 = RWKV(base_dim*4,4,8,0,RLN1D)
        self.level3_block2 = DEABlockTrain(default_conv, base_dim * 4, 3)
        # self.level3_RWKV2 = RWKV(base_dim*4,4,8,1,RLN1D)
        # self.level3_block3 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_RWKV3 = RWKV(base_dim*4,4,6,0,RLN1D)
        # self.level3_block4 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_RWKV4 = RWKV(base_dim*4,4,6,1,RLN1D)
        # self.level3_block5 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_RWKV5 = RWKV(base_dim*4,4,6,2,RLN1D)
        # self.level3_block6 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_RWKV6 = RWKV(base_dim*4,4,6,3,RLN1D)
        # self.level3_block7 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_RWKV7 = RWKV(base_dim*4,4,6,4,RLN1D)
        # self.level3_block8 = DEABlockTrain(default_conv, base_dim * 4, 3)
        self.level3_RWKV8 = RWKV(base_dim*4,4,6,5,RLN1D)
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
        # x_down2_init = self.down_level2_block3(x_down2_init)
        # x_down2_init = self.down_level2_block4(x_down2_init)
        x_down2_init = self.level2_RWKV1(x_down2_init)
        x_down2_init = self.level2_RWKV2(x_down2_init)


        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)

        x1 = self.level3_block1(x_down3_init)
        # x1_1 = self.level3_RWKV1(x1)

        x2 = self.level3_block2(x1)
        # x2_1 = self.level3_RWKV2(x2)
        
        # x3 = self.level3_block3(x2_1)
        x3_1 = self.level3_RWKV3(x2)

        # x4 = self.level3_block4(x3_1)
        x4_1 = self.level3_RWKV4(x3_1)

        # x5 = self.level3_block5(x4_1)
        x5_1 = self.level3_RWKV5(x4_1)

        # x6 = self.level3_block6(x5_1)
        x6_1 = self.level3_RWKV6(x5_1)

        # x7 = self.level3_block7(x6_1)
        x7_1 = self.level3_RWKV7(x6_1)

        # x8 = self.level3_block8(x7_1)
        x8_1 = self.level3_RWKV8(x7_1)

        x_level3_mix = self.mix1(x_down3, x8_1)

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