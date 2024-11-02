from models import *
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

class ParallelU3UNet(ISNetDIS):
    def __init__(self, *args, **kwargs):
        super(ParallelU3UNet, self).__init__(
            in_ch=3, out_ch=1)

        self.seq1 = nn.Sequential(
            self.conv_in,
            self.pool_in,
            self.stage1,
            self.pool12,

            self.stage2,
            self.pool23,
            self.stage3,
            self.pool34,
            self.stage4,
            self.pool45,
            self.stage5,
            self.pool56,
            self.stage6,

            # self.pool1to5d,
            self.stage1to5d,
            self.pool2to5d,
            self.stage2to5d,
            # 分辨率：128 -> 32
            self.pool3to5d,
            self.stage3to5d,
            # 分辨率：64 -> 32
            self.pool4to5d,
            self.stage4to5d,
            # 分辨率：32 -> 32
            self.stage5to5d,
            # 分辨率：16 -> 32
            self.up6to5d,
            self.stage6to5d,
            # concat -> rsu
            self.stage5,
            self.pool1to4d,
            self.stage1to4d,
            # 分辨率：256 -> 64
            self.pool2to4d,
            self.stage2to4d,
            # 分辨率：128 -> 64
            self.pool3to4d,
            self.stage3to4d,
            # 分辨率：64 -> 64
            self.stage4to4d,
            # 分辨率：16 -> 64
            self.up6to4d,
            self.stage6to4d,
            # 分辨率：32 -> 64
            self.up5dto4d,
            self.stage5dto4d,
            # concat -> rsu
            self.stage4d,
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.pool1to3d,
            self.stage1to3d,
            # 分辨率：256 -> 128
            self.pool2to3d,
            self.stage2to3d,
            # 分辨率：128 -> 128
            self.stage3to3d,
            # 分辨率：16 -> 128
            self.up6to3d,
            self.stage6to3d,
            # 分辨率：32 -> 128
            self.up5dto3d,
            self.stage5dto3d,
            # 分辨率：64 -> 128
            self.up4dto3d,
            self.stage4dto3d,
            # concat -> rsu
            self.stage3d,
            self.pool1to2d,
            self.stage1to2d,
            # 分辨率：256 -> 256
            self.stage2to2d,
            # 分辨率：16 -> 256
            self.up6to2d,
            self.stage6to2d,
            # 分辨率：32 -> 256
            self.up5dto2d,
            self.stage5dto2d,
            # 分辨率：64 -> 256
            self.up4dto2d,
            self.stage4dto2d,
            # 分辨率：128 -> 256
            self.up3dto2d,
            self.stage3dto2d,
            # concat -> rsu
            self.stage2d,
            self.stage1to1d,
            # 分辨率：16 -> 512
            self.up6to1d,
            self.stage6to1d,
            # 分辨率：32 -> 512
            self.up5dto1d,
            self.stage5dto1d,
            # 分辨率：64 -> 512
            self.up4dto1d,
            self.stage4dto1d,
            # 分辨率：128 -> 512
            self.up3dto1d,
            self.stage3dto1d,
            # 分辨率：256 -> 512
            self.up2dto1d,
            self.stage2dto1d,
            # concat -> rsu
            self.stage1d,
            self.side1,
            self.side2,
            self.side3,
            self.side4,
            self.side5,
            self.side6
        ).to('cuda:1')


    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return x
