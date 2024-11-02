import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

bce_loss = nn.BCELoss(size_average=True)


def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        # print("i: ", i, preds[i].shape)
        if (preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if (i == 0):
            loss0 = loss
    return loss0, loss


fea_loss = nn.MSELoss(size_average=True)
kl_loss = nn.KLDivLoss(size_average=True)
l1_loss = nn.L1Loss(size_average=True)
smooth_l1_loss = nn.SmoothL1Loss(size_average=True)


def muti_loss_fusion_kl(preds, target, dfs, fs, mode='MSE'):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        # print("i: ", i, preds[i].shape)
        if (preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if (i == 0):
            loss0 = loss

    for i in range(0, len(dfs)):
        if (mode == 'MSE'):
            loss = loss + fea_loss(dfs[i], fs[i])  ### add the mse loss of features as additional constraints
            # print("fea_loss: ", fea_loss(dfs[i],fs[i]).item())
        elif (mode == 'KL'):
            loss = loss + kl_loss(F.log_softmax(dfs[i], dim=1), F.softmax(fs[i], dim=1))
            # print("kl_loss: ", kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1)).item())
        elif (mode == 'MAE'):
            loss = loss + l1_loss(dfs[i], fs[i])
            # print("ls_loss: ", l1_loss(dfs[i],fs[i]))
        elif (mode == 'SmoothL1'):
            loss = loss + smooth_l1_loss(dfs[i], fs[i])
            # print("SmoothL1: ", smooth_l1_loss(dfs[i],fs[i]).item())

    return loss0, loss


class REBNCONV(nn.Module):
    # flag = 1
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # print("flag: " + str(REBNCONV.flag) + '\n')
        # REBNCONV.flag += 1
        print(x.shape)
        hx = x
        c = self.conv_s1(hx)
        b = self.bn_s1(c)
        xout = self.relu_s1(b)
        return xout


class REBNCONV1(nn.Module):
    # flag = 1
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV1, self).__init__()
        self.to('cuda:1')
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # print("flag: " + str(REBNCONV.flag) + '\n')
        # REBNCONV.flag += 1
        print(x.shape)
        hx = x
        # print(self.conv_s1.is_cuda, hx.is_cuda)
        c = self.conv_s1(hx)
        b = self.bn_s1(c)
        xout = self.relu_s1(b)
        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):  # 64 32 64
        super(RSU7, self).__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  ## 1 -> 1/2

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        print("7-start")
        b, c, h, w = x.shape

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        final_x = hx1d + hxin
        print("final_x")
        print(final_x.shape)
        print("7-end")
        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        print("6-start")
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        print("6-end")
        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        print("5-start")
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        print("5-end")
        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        print("4-start")
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        print("4-end")
        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        print("4F-start")
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        print("4F-end")
        return hx1d + hxin


class Stage5d(nn.Module):

    def __init__(self):
        super(Stage5d, self).__init__()

        ## stage5d
        # 分辨率：512 -> 32
        self.pool1to5d = nn.MaxPool2d(16, 16, ceil_mode=True).to('cuda:1')
        self.stage1to5d = REBNCONV1(64, 64, dirate=1).to('cuda:1')

        # 分辨率：256 -> 32
        self.pool2to5d = nn.MaxPool2d(8, 8, ceil_mode=True).to('cuda:1')
        self.stage2to5d = REBNCONV1(128, 64, dirate=1).to('cuda:1')

        # 分辨率：128 -> 32
        self.pool3to5d = nn.MaxPool2d(4, 4, ceil_mode=True).to('cuda:1')
        self.stage3to5d = REBNCONV1(256, 64, dirate=1).to('cuda:1')

        # 分辨率：64 -> 32
        self.pool4to5d = nn.MaxPool2d(2, 2, ceil_mode=True).to('cuda:1')
        self.stage4to5d = REBNCONV1(512, 64, dirate=1).to('cuda:1')

        # 分辨率：32 -> 32
        self.stage5to5d = REBNCONV1(512, 64, dirate=1).to('cuda:1')

        # 分辨率：16 -> 32
        self.up6to5d = nn.Upsample(scale_factor=2, mode='bilinear').to('cuda:1')
        self.stage6to5d = REBNCONV1(512, 64, dirate=1).to('cuda:1')

        # concat -> rsu
        self.stage5d = RSU4F(384, 192, 384)
        ##

    def forward(self, hx1, hx2, hx3, hx4, hx5, hx6):
        ## stage5d
        # 分辨率：512 -> 32
        hx1to5d = self.stage1to5d(self.pool1to5d(hx1.to('cuda:1'))).to('cuda:1')
        # 分辨率：256 -> 32
        hx2to5d = self.stage2to5d(self.pool2to5d(hx2)).to('cuda:1')

        # 分辨率：128 -> 32
        hx3to5d = self.stage3to5d(self.pool3to5d(hx3)).to('cuda:1')

        # 分辨率：64 -> 32
        hx4to5d = self.stage4to5d(self.pool4to5d(hx4)).to('cuda:1')

        # 分辨率：32 -> 32
        hx5to5d = self.stage5to5d(hx5).to('cuda:1')

        # 分辨率：16 -> 32
        hx6to5d = self.stage6to5d(self.up6to5d(hx6)).to('cuda:1')

        hx5d = self.stage5d(torch.cat((hx1to5d, hx2to5d, hx3to5d, hx4to5d, hx5to5d, hx6to5d), 1))
        ##

        return hx5d


class Stage4d(nn.Module):

    def __init__(self):
        super(Stage4d, self).__init__()

        ## stage4d
        # 分辨率：512 -> 64
        self.pool1to4d = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.stage1to4d = REBNCONV(64, 64, dirate=1)

        # 分辨率：256 -> 64
        self.pool2to4d = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.stage2to4d = REBNCONV(128, 64, dirate=1)

        # 分辨率：128 -> 64
        self.pool3to4d = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage3to4d = REBNCONV(256, 64, dirate=1)

        # 分辨率：64 -> 64
        self.stage4to4d = REBNCONV(512, 64, dirate=1)

        # 分辨率：16 -> 64
        self.up6to4d = nn.Upsample(scale_factor=4, mode='bilinear')
        self.stage6to4d = REBNCONV(512, 64, dirate=1)

        # 分辨率：32 -> 64
        self.up5dto4d = nn.Upsample(scale_factor=2, mode='bilinear')
        self.stage5dto4d = REBNCONV(384, 64, dirate=1)

        # concat -> rsu
        self.stage4d = RSU4(384, 192, 384)
        ##

    def forward(self, hx1, hx2, hx3, hx4, hx6, hx5d):
        ## stage4d
        # 分辨率：512 -> 64
        hx1to4d = self.stage1to4d(self.pool1to4d(hx1))

        # 分辨率：256 -> 64
        hx2to4d = self.stage2to4d(self.pool2to4d(hx2))

        # 分辨率：128 -> 64
        hx3to4d = self.stage3to4d(self.pool3to4d(hx3))

        # 分辨率：64 -> 64
        hx4to4d = self.stage4to4d(hx4)

        # 分辨率：16 -> 64
        hx6to4d = self.stage6to4d(self.up6to4d(hx6))

        # 分辨率：32 -> 64
        hx5dto4d = self.stage5dto4d(self.up5dto4d(hx5d))

        hx4d = self.stage4d(torch.cat((hx1to4d, hx2to4d, hx3to4d, hx4to4d, hx6to4d, hx5dto4d), 1))
        ##

        return hx4d


class Stage3d(nn.Module):

    def __init__(self):
        super(Stage3d, self).__init__()

        ## stage3d
        # 分辨率：512 -> 128
        self.pool1to3d = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.stage1to3d = REBNCONV(64, 64, dirate=1)

        # 分辨率：256 -> 128
        self.pool2to3d = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage2to3d = REBNCONV(128, 64, dirate=1)

        # 分辨率：128 -> 128
        self.stage3to3d = REBNCONV(256, 64, dirate=1)

        # 分辨率：16 -> 128
        self.up6to3d = nn.Upsample(scale_factor=8, mode='bilinear')
        self.stage6to3d = REBNCONV(512, 64, dirate=1)

        # 分辨率：32 -> 128
        self.up5dto3d = nn.Upsample(scale_factor=4, mode='bilinear')
        self.stage5dto3d = REBNCONV(384, 64, dirate=1)

        # 分辨率：64 -> 128
        self.up4dto3d = nn.Upsample(scale_factor=2, mode='bilinear')
        self.stage4dto3d = REBNCONV(384, 64, dirate=1)

        # concat -> rsu
        self.stage3d = RSU5(384, 192, 384)
        ##

    def forward(self, hx1, hx2, hx3, hx6, hx5d, hx4d):
        ## stage3d
        # 分辨率：512 -> 128
        hx1to3d = self.stage1to3d(self.pool1to3d(hx1))

        # 分辨率：256 -> 128
        hx2to3d = self.stage2to3d(self.pool2to3d(hx2))

        # 分辨率：128 -> 128
        hx3to3d = self.stage3to3d(hx3)

        # 分辨率：16 -> 128
        hx6to3d = self.stage6to3d(self.up6to3d(hx6))

        # 分辨率：32 -> 128
        hx5dto3d = self.stage5dto3d(self.up5dto3d(hx5d))

        # 分辨率：64 -> 128
        hx4dto3d = self.stage4dto3d(self.up4dto3d(hx4d))

        hx3d = self.stage3d(torch.cat((hx1to3d, hx2to3d, hx3to3d, hx6to3d, hx5dto3d, hx4dto3d), 1))
        ##

        return hx3d


class Stage2d(nn.Module):

    def __init__(self):
        super(Stage2d, self).__init__()

        ## stage2d
        # 分辨率：512 -> 256
        self.pool1to2d = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage1to2d = REBNCONV(64, 64, dirate=1)

        # 分辨率：256 -> 256
        self.stage2to2d = REBNCONV(128, 64, dirate=1)

        # 分辨率：16 -> 256
        self.up6to2d = nn.Upsample(scale_factor=16, mode='bilinear')
        self.stage6to2d = REBNCONV(512, 64, dirate=1)

        # 分辨率：32 -> 256
        self.up5dto2d = nn.Upsample(scale_factor=8, mode='bilinear')
        self.stage5dto2d = REBNCONV(384, 64, dirate=1)

        # 分辨率：64 -> 256
        self.up4dto2d = nn.Upsample(scale_factor=4, mode='bilinear')
        self.stage4dto2d = REBNCONV(384, 64, dirate=1)

        # 分辨率：128 -> 256
        self.up3dto2d = nn.Upsample(scale_factor=2, mode='bilinear')
        self.stage3dto2d = REBNCONV(384, 64, dirate=1)

        # concat -> rsu
        self.stage2d = RSU6(384, 192, 384)
        ##

    def forward(self, hx1, hx2, hx6, hx5d, hx4d, hx3d):
        ## stage2d
        # 分辨率：512 -> 256
        hx1to2d = self.stage1to2d(self.pool1to2d(hx1))

        # 分辨率：256 -> 256
        hx2to2d = self.stage2to2d(hx2)

        # 分辨率：16 -> 256
        hx6to2d = self.stage6to2d(self.up6to2d(hx6))

        # 分辨率：32 -> 256
        hx5dto2d = self.stage5dto2d(self.up5dto2d(hx5d))

        # 分辨率：64 -> 256
        hx4dto2d = self.stage4dto2d(self.up4dto2d(hx4d))

        # 分辨率：128 -> 256
        hx3dto2d = self.stage3dto2d(self.up3dto2d(hx3d))

        hx2d = self.stage2d(torch.cat((hx1to2d, hx2to2d, hx6to2d, hx5dto2d, hx4dto2d, hx3dto2d), 1))
        ##

        return hx2d


class Stage1d(nn.Module):

    def __init__(self):
        super(Stage1d, self).__init__()

        ## stage1d
        # 分辨率：512 -> 512
        self.stage1to1d = REBNCONV(64, 64, dirate=1)

        # 分辨率：16 -> 512
        self.up6to1d = nn.Upsample(scale_factor=32, mode='bilinear')
        self.stage6to1d = REBNCONV(512, 64, dirate=1)

        # 分辨率：32 -> 512
        self.up5dto1d = nn.Upsample(scale_factor=16, mode='bilinear')
        self.stage5dto1d = REBNCONV(384, 64, dirate=1)

        # 分辨率：64 -> 512
        self.up4dto1d = nn.Upsample(scale_factor=8, mode='bilinear')
        self.stage4dto1d = REBNCONV(384, 64, dirate=1)

        # 分辨率：128 -> 512
        self.up3dto1d = nn.Upsample(scale_factor=4, mode='bilinear')
        self.stage3dto1d = REBNCONV(384, 64, dirate=1)

        # 分辨率：256 -> 512
        self.up2dto1d = nn.Upsample(scale_factor=2, mode='bilinear')
        self.stage2dto1d = REBNCONV(384, 64, dirate=1)

        # concat -> rsu
        self.stage1d = RSU7(384, 192, 384)
        ##

    def forward(self, hx1, hx6, hx5d, hx4d, hx3d, hx2d):
        ## stage1d
        # 分辨率：512 -> 512
        hx1to1d = self.stage1to1d(hx1)

        # 分辨率：16 -> 512
        hx6to1d = self.stage6to1d(self.up6to1d(hx6))

        # 分辨率：32 -> 512
        hx5dto1d = self.stage5dto1d(self.up5dto1d(hx5d))

        # 分辨率：64 -> 512
        hx4dto1d = self.stage4dto1d(self.up4dto1d(hx4d))

        # 分辨率：128 -> 512
        hx3dto1d = self.stage3dto1d(self.up3dto1d(hx3d))

        # 分辨率：256 -> 512
        hx2dto1d = self.stage2dto1d(self.up2dto1d(hx2d))

        hx1d = self.stage1d(torch.cat((hx1to1d, hx6to1d, hx5dto1d, hx4dto1d, hx3dto1d, hx2dto1d), 1))
        ##

        return hx1d


class myrebnconv(nn.Module):
    def __init__(self, in_ch=3,
                 out_ch=1,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1):
        super(myrebnconv, self).__init__()

        self.conv = nn.Conv2d(in_ch,
                              out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.rl(self.bn(self.conv(x)))


class ISNetGTEncoder(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(ISNetGTEncoder, self).__init__()

        self.conv_in = myrebnconv(in_ch, 16, 3, stride=2, padding=1)  # nn.Conv2d(in_ch,64,3,stride=2,padding=1)

        self.stage1 = RSU7(16, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(128, 32, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 64, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 64, 512)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def compute_loss(self, preds, targets):
        return muti_loss_fusion(preds, targets)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        # side output
        d1 = self.side1(hx1)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)], [hx1, hx2,
                                                                                                            hx3, hx4,
                                                                                                            hx5, hx6]


class ISNetDIS(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(ISNetDIS, self).__init__()

        ### encoder
        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        # self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # downsample is same as max-pooling

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        ### decoder

        # ## stage5d
        # # 分辨率：512 -> 32
        # self.pool1to5d = nn.MaxPool2d(16, 16, ceil_mode=True)
        # self.stage1to5d = REBNCONV(64, 64, dirate=1)
        #
        # # 分辨率：256 -> 32
        # self.pool2to5d = nn.MaxPool2d(8, 8, ceil_mode=True)
        # self.stage2to5d = REBNCONV(128, 64, dirate=1)
        #
        # # 分辨率：128 -> 32
        # self.pool3to5d = nn.MaxPool2d(4, 4, ceil_mode=True)
        # self.stage3to5d = REBNCONV(256, 64, dirate=1)
        #
        # # 分辨率：64 -> 32
        # self.pool4to5d = nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.stage4to5d = REBNCONV(512, 64, dirate=1)
        #
        # # 分辨率：32 -> 32
        # self.stage5to5d = REBNCONV(512, 64, dirate=1)
        #
        # # 分辨率：16 -> 32
        # self.up6to5d = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.stage6to5d = REBNCONV(512, 64, dirate=1)
        #
        # # concat -> rsu
        # self.stage5d = RSU4F(384, 192, 384)
        # ##

        # ## stage4d
        # # 分辨率：512 -> 64
        # self.pool1to4d = nn.MaxPool2d(8, 8, ceil_mode=True)
        # self.stage1to4d = REBNCONV(64, 64, dirate=1)
        #
        # # 分辨率：256 -> 64
        # self.pool2to4d = nn.MaxPool2d(4, 4, ceil_mode=True)
        # self.stage2to4d = REBNCONV(128, 64, dirate=1)
        #
        # # 分辨率：128 -> 64
        # self.pool3to4d = nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.stage3to4d = REBNCONV(256, 64, dirate=1)
        #
        # # 分辨率：64 -> 64
        # self.stage4to4d = REBNCONV(512, 64, dirate=1)
        #
        # # 分辨率：16 -> 64
        # self.up6to4d = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.stage6to4d = REBNCONV(512, 64, dirate=1)
        #
        # # 分辨率：32 -> 64
        # self.up5dto4d = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.stage5dto4d = REBNCONV(384, 64, dirate=1)
        #
        # # concat -> rsu
        # self.stage4d = RSU4(384, 192, 384)
        # ##

        # ## stage3d
        # # 分辨率：512 -> 128
        # self.pool1to3d = nn.MaxPool2d(4, 4, ceil_mode=True)
        # self.stage1to3d = REBNCONV(64, 64, dirate=1)
        #
        # # 分辨率：256 -> 128
        # self.pool2to3d = nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.stage2to3d = REBNCONV(128, 64, dirate=1)
        #
        # # 分辨率：128 -> 128
        # self.stage3to3d = REBNCONV(256, 64, dirate=1)
        #
        # # 分辨率：16 -> 128
        # self.up6to3d = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.stage6to3d = REBNCONV(512, 64, dirate=1)
        #
        # # 分辨率：32 -> 128
        # self.up5dto3d = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.stage5dto3d = REBNCONV(384, 64, dirate=1)
        #
        # # 分辨率：64 -> 128
        # self.up4dto3d = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.stage4dto3d = REBNCONV(384, 64, dirate=1)
        #
        # # concat -> rsu
        # self.stage3d = RSU5(384, 192, 384)
        # ##

        # ## stage2d
        # # 分辨率：512 -> 256
        # self.pool1to2d = nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.stage1to2d = REBNCONV(64, 64, dirate=1)
        #
        # # 分辨率：256 -> 256
        # self.stage2to2d = REBNCONV(128, 64, dirate=1)
        #
        # # 分辨率：16 -> 256
        # self.up6to2d = nn.Upsample(scale_factor=16, mode='bilinear')
        # self.stage6to2d = REBNCONV(512, 64, dirate=1)
        #
        # # 分辨率：32 -> 256
        # self.up5dto2d = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.stage5dto2d = REBNCONV(384, 64, dirate=1)
        #
        # # 分辨率：64 -> 256
        # self.up4dto2d = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.stage4dto2d = REBNCONV(384, 64, dirate=1)
        #
        # # 分辨率：128 -> 256
        # self.up3dto2d = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.stage3dto2d = REBNCONV(384, 64, dirate=1)
        #
        # # concat -> rsu
        # self.stage2d = RSU6(384, 192, 384)
        # ##

        # ## stage1d
        # # 分辨率：512 -> 512
        # self.stage1to1d = REBNCONV(64, 64, dirate=1)
        #
        # # 分辨率：16 -> 512
        # self.up6to1d = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.stage6to1d = REBNCONV(512, 64, dirate=1)
        #
        # # 分辨率：32 -> 512
        # self.up5dto1d = nn.Upsample(scale_factor=16, mode='bilinear')
        # self.stage5dto1d = REBNCONV(384, 64, dirate=1)
        #
        # # 分辨率：64 -> 512
        # self.up4dto1d = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.stage4dto1d = REBNCONV(384, 64, dirate=1)
        #
        # # 分辨率：128 -> 512
        # self.up3dto1d = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.stage3dto1d = REBNCONV(384, 64, dirate=1)
        #
        # # 分辨率：256 -> 512
        # self.up2dto1d = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.stage2dto1d = REBNCONV(384, 64, dirate=1)
        #
        # # concat -> rsu
        # self.stage1d = RSU7(384, 192, 384)
        # ##

        self.stage5dfunc = Stage5d().to('cuda:1')
        self.stage4dfunc = Stage4d().to('cuda:1')
        self.stage3dfunc = Stage3d().to('cuda:1')
        self.stage2dfunc = Stage2d().to('cuda:1')
        self.stage1dfunc = Stage1d().to('cuda:1')

        self.side1 = nn.Conv2d(384, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(384, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(384, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(384, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(384, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def compute_loss_kl(self, preds, targets, dfs, fs, mode='MSE'):
        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion_kl(preds, targets, dfs, fs, mode=mode)

    def compute_loss(self, preds, targets):
        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion(preds, targets)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # -------------------- encoder --------------------
        # stage 1

        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        # -------------------- decoder --------------------

        # ## stage5d
        # # 分辨率：512 -> 32
        # hx1to5d = self.stage1to5d(self.pool1to5d(hx1))
        # # 分辨率：256 -> 32
        # hx2to5d = self.stage2to5d(self.pool2to5d(hx2))
        #
        # # 分辨率：128 -> 32
        # hx3to5d = self.stage3to5d(self.pool3to5d(hx3))
        #
        # # 分辨率：64 -> 32
        # hx4to5d = self.stage4to5d(self.pool4to5d(hx4))
        #
        # # 分辨率：32 -> 32
        # hx5to5d = self.stage5to5d(hx5)
        #
        # # 分辨率：16 -> 32
        # hx6to5d = self.stage6to5d(self.up6to5d(hx6))
        #
        # hx5d = self.stage5d(torch.cat((hx1to5d, hx2to5d, hx3to5d, hx4to5d, hx5to5d, hx6to5d), 1))
        # ##
        hx5d = self.stage5dfunc(hx1.to('cuda:1'), hx2.to('cuda:1'), hx3.to('cuda:1'), hx4.to('cuda:1'),
                                hx5.to('cuda:1'), hx6.to('cuda:1')).to('cuda:1')

        # ## stage4d
        # # 分辨率：512 -> 64
        # hx1to4d = self.stage1to4d(self.pool1to4d(hx1))
        #
        # # 分辨率：256 -> 64
        # hx2to4d = self.stage2to4d(self.pool2to4d(hx2))
        #
        # # 分辨率：128 -> 64
        # hx3to4d = self.stage3to4d(self.pool3to4d(hx3))
        #
        # # 分辨率：64 -> 64
        # hx4to4d = self.stage4to4d(hx4)
        #
        # # 分辨率：16 -> 64
        # hx6to4d = self.stage6to4d(self.up6to4d(hx6))
        #
        # # 分辨率：32 -> 64
        # hx5dto4d = self.stage5dto4d(self.up5dto4d(hx5d))
        #
        # hx4d = self.stage4d(torch.cat((hx1to4d, hx2to4d, hx3to4d, hx4to4d, hx6to4d, hx5dto4d), 1))
        # ##
        hx4d = self.stage4dfunc(hx1.to('cuda:1'), hx2.to('cuda:1'), hx3.to('cuda:1'), hx4.to('cuda:1'),
                                hx6.to('cuda:1'), hx5d.to('cuda:1'))

        # ## stage3d
        # # 分辨率：512 -> 128
        # hx1to3d = self.stage1to3d(self.pool1to3d(hx1))
        #
        # # 分辨率：256 -> 128
        # hx2to3d = self.stage2to3d(self.pool2to3d(hx2))
        #
        # # 分辨率：128 -> 128
        # hx3to3d = self.stage3to3d(hx3)
        #
        # # 分辨率：16 -> 128
        # hx6to3d = self.stage6to3d(self.up6to3d(hx6))
        #
        # # 分辨率：32 -> 128
        # hx5dto3d = self.stage5dto3d(self.up5dto3d(hx5d))
        #
        # # 分辨率：64 -> 128
        # hx4dto3d = self.stage4dto3d(self.up4dto3d(hx4d))
        #
        # hx3d = self.stage3d(torch.cat((hx1to3d, hx2to3d, hx3to3d, hx6to3d, hx5dto3d, hx4dto3d), 1))
        # ##
        hx3d = self.stage3dfunc(hx1.to('cuda:1'), hx2.to('cuda:1'), hx3.to('cuda:1'), hx6.to('cuda:1'),
                                hx5d.to('cuda:1'), hx4d.to('cuda:1'))

        # ## stage2d
        # # 分辨率：512 -> 256
        # hx1to2d = self.stage1to2d(self.pool1to2d(hx1))
        #
        # # 分辨率：256 -> 256
        # hx2to2d = self.stage2to2d(hx2)
        #
        # # 分辨率：16 -> 256
        # hx6to2d = self.stage6to2d(self.up6to2d(hx6))
        #
        # # 分辨率：32 -> 256
        # hx5dto2d = self.stage5dto2d(self.up5dto2d(hx5d))
        #
        # # 分辨率：64 -> 256
        # hx4dto2d = self.stage4dto2d(self.up4dto2d(hx4d))
        #
        # # 分辨率：128 -> 256
        # hx3dto2d = self.stage3dto2d(self.up3dto2d(hx3d))
        #
        # hx2d = self.stage2d(torch.cat((hx1to2d, hx2to2d, hx6to2d, hx5dto2d, hx4dto2d, hx3dto2d), 1))
        # ##
        hx2d = self.stage2dfunc(hx1.to('cuda:1'), hx2.to('cuda:1'), hx6.to('cuda:1'), hx5d.to('cuda:1'),
                                hx4d.to('cuda:1'), hx3d.to('cuda:1'))

        # ## stage1d
        # # 分辨率：512 -> 512
        # hx1to1d = self.stage1to1d(hx1)
        #
        # # 分辨率：16 -> 512
        # hx6to1d = self.stage6to1d(self.up6to1d(hx6))
        #
        # # 分辨率：32 -> 512
        # hx5dto1d = self.stage5dto1d(self.up5dto1d(hx5d))
        #
        # # 分辨率：64 -> 512
        # hx4dto1d = self.stage4dto1d(self.up4dto1d(hx4d))
        #
        # # 分辨率：128 -> 512
        # hx3dto1d = self.stage3dto1d(self.up3dto1d(hx3d))
        #
        # # 分辨率：256 -> 512
        # hx2dto1d = self.stage2dto1d(self.up2dto1d(hx2d))
        #
        # hx1d = self.stage1d(torch.cat((hx1to1d, hx6to1d, hx5dto1d, hx4dto1d, hx3dto1d, hx2dto1d), 1))
        # ##
        hx1d = self.stage1dfunc(hx1.to('cuda:1'), hx6.to('cuda:1'), hx5d.to('cuda:1'), hx4d.to('cuda:1'),
                                hx3d.to('cuda:1'), hx2d.to('cuda:1'))
        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)], [hx1d, hx2d,
                                                                                                            hx3d, hx4d,
                                                                                                            hx5d, hx6]
