import math
import torch
import torch.nn as nn
import torchvision.models as models
from sympy.codegen import Print
from lib.pvt import pvt_v2_b2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import DeformConv2d
import torchvision.ops


class DFD(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        # 方向感知卷积
        self.directional_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel),
            nn.Conv2d(channel, channel // 2, 1)
        )

        # 周期性模式捕捉
        self.periodic_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // 2, channel // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel // 2, 1),
            nn.Sigmoid()
        )

        # 跨模态增强
        self.cross_modal = nn.Conv2d(channel // 2 * 2, channel, 1)

    def forward(self, img, depth):
        # 方向特征提取
        dir_img = self.directional_conv(img)
        dir_depth = self.directional_conv(depth)
        # 周期性增强
        att_img = dir_img * self.periodic_att(dir_img)
        att_depth = dir_depth * self.periodic_att(dir_depth)
        # 跨模态融合
        return self.cross_modal(torch.cat([att_img, att_depth], dim=1)) + img + depth


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18], out_channels=32):
        super(ASPP, self).__init__()
        modules = []

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SMAR(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.dfd_att = DFD(channel)
        self.hallucination_gen = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 3, padding=1),
            nn.InstanceNorm2d(channel // 4),
            nn.GELU(),
            nn.Conv2d(channel // 4, channel, 1)
        )

    def forward(self, img, depth):
        att_feat = self.dfd_att(img, depth)
        # 生成幻觉特征
        hallucination = self.hallucination_gen(att_feat)
        # 特征校正
        return att_feat + hallucination * (1 - att_feat.sigmoid())


# Guidance-based Gated Attention
class GGA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(GGA, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels+1,  1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out_cov = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, in_feat, gate_feat):
        attention_map = self.gate_conv(torch.cat([in_feat, gate_feat], dim=1))
        in_feat = (in_feat * (attention_map + 1))
        out_feat = self.out_cov(in_feat)
        return out_feat


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):  # act = ReLU or PReLU
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# Residual Feature Decoder
class RFD(nn.Module):
    def __init__(self, channel, kernel_size, reduction, bias, act, n_resblocks):
        super(RFD, self).__init__()
        modules_body = [RCAB(channel, kernel_size, reduction, bias=bias, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(channel, channel, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = torchvision.ops.deform_conv2d(x, offset, self.conv.weight, self.conv.bias, padding=1)
        return x

class DynamicInteractionUnit(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.mlp(x)

class CUP(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.level_convs = nn.ModuleList()
        current_channels = in_channels

        for _ in range(num_levels):
            self.level_convs.append(
                nn.Sequential(
                    DeformableConv2d(current_channels, out_channels),
                    DynamicInteractionUnit(out_channels)
                )
            )
            current_channels = out_channels

        self.fuse_conv = nn.Conv2d(num_levels * out_channels, out_channels, 1)

    def forward(self, x):
        features = []
        current_x = x
        original_size = x.shape[2:]  # 记录初始尺寸 [H, W]

        for i, conv in enumerate(self.level_convs):
            if i > 0:
                current_x = F.interpolate(current_x, scale_factor=0.5, mode='bilinear', align_corners=False)
            current_x = conv(current_x)
            # 将当前层输出插值回初始尺寸
            resized_x = F.interpolate(current_x, size=original_size, mode='bilinear', align_corners=False)
            features.append(resized_x)

        # 自底向上融合（此时所有特征图尺寸一致）
        for i in range(self.num_levels - 2, -1, -1):
            features[i] = features[i] + features[i + 1]

        fused = torch.cat(features, dim=1)
        return self.fuse_conv(fused)

class DFNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3, reduction=4, bias=False, act=nn.PReLU(), n_resblocks=2, iteration=3):
        super(DFNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r'E:\PythonProjectbs\DFNet\pvt_v2_b2.pth'
        # save_model = torch.load(path)
        save_model = torch.load(path, map_location=torch.device('cuda'),weights_only=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.iteration = iteration


        self.CUP_4 = CUP(64, 32, num_levels=2)
        self.CUP_3 = CUP(128, 32, num_levels=2)
        self.CUP_2 = CUP(320, 32, num_levels=2)
        self.CUP_1 = CUP(512, 32, num_levels=2)

        self.smar_4 = SMAR(channel)
        self.smar_3 = SMAR(channel)
        self.smar_2 = SMAR(channel)
        self.smar_1 = SMAR(channel)

        self.gate_1 = GGA(channel,channel)
        self.gate_2 = GGA(channel,channel)
        self.gate_3 = GGA(channel,channel)



        self.rfd_1 = RFD(channel, kernel_size, reduction, bias, act, n_resblocks)  # 32 x 22 x 22
        self.rfd_2 = RFD(2 * channel, kernel_size, reduction, bias, act, n_resblocks)  # 64 x 44 x 44
        self.rfd_3 = RFD(3 * channel, kernel_size, reduction, bias, act, n_resblocks)  # 96 x 88 x 88



        self.gate_conv = nn.Sequential(
            BasicConv2d(32, 1, 1),
            nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        )
        self.gate_conv_1 = BasicConv2d(32, 1, 1)
        self.gate_conv_2 = BasicConv2d(64, 1, 1)

        self.unsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.pred = nn.Conv2d(channel, 1, 1)

        self.Fus = ASPP(2 * channel)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_pred = nn.Conv2d(channel, 1, 1)

    def forward(self, x):  # 孪生网络，这里是batch拼接后的img和depth

        pvt = self.backbone(x)
        x4 = pvt[0]  # 64x176x176
        x3 = pvt[1]  # 128x88x88
        x2 = pvt[2]  # 320x44x44
        x1 = pvt[3]  # 512x22x22

        x4 = self.CUP_4(x4)
        x3 = self.CUP_3(x3)
        x2 = self.CUP_2(x2)
        x1 = self.CUP_1(x1)

        x4_img, x4_depth = torch.chunk(x4, 2, dim=0)
        x3_img, x3_depth = torch.chunk(x3, 2, dim=0)
        x2_img, x2_depth = torch.chunk(x2, 2, dim=0)
        x1_img, x1_depth = torch.chunk(x1, 2, dim=0)

        stage_pred = list()
        coarse_pred = None  # 阶段预测结果
        for iter in range(self.iteration):
            x1 = self.smar_1(x1_img, x1_depth)
            if coarse_pred == None:
                x1 = x1
            else:
                coarse_pred = self.gate_conv(coarse_pred)
                x1 = self.gate_1(x1, coarse_pred)
            x2_feed = self.rfd_1(x1)
            x2 = self.smar_2(x2_img, x2_depth)
            if iter > 0:
                x2_gate = self.unsample_2(self.gate_conv_1(x2_feed))
                x2 = self.gate_2(x2, x2_gate)
            x3_feed = self.rfd_2(torch.cat((x2, self.unsample_2(x2_feed)), dim=1))
            x3 = self.smar_3(x3_img, x3_depth)
            if iter > 0:
                x3_gate = self.unsample_2(self.gate_conv_2(x3_feed))
                x3 = self.gate_3(x3, x3_gate)
            x4_feed = self.rfd_3(torch.cat((x3, self.unsample_2(x3_feed)), dim=1))
            coarse_pred = self.out(x4_feed)
            out_map = self.pred(coarse_pred)
            pred = F.interpolate(out_map, scale_factor=8, mode='bilinear')
            stage_pred.append(pred)

        x4 = self.smar_4(x4_img, x4_depth)
        x4_out = self.downsample(x4)
        x_in = torch.cat((coarse_pred, x4_out), dim=1)
        refined_pred = self.Fus(x_in)

        pred2 = self.out_pred(refined_pred)
        final_pred = F.interpolate(pred2, scale_factor=8, mode='bilinear')
        return stage_pred, final_pred

# model = RISNet()
# print(model)










