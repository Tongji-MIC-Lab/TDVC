import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from main.utils.dcnv2.dcn_v2 import DCN as ConvOffset2d
from main.utils.dcnv2.dcn_v2_amp import DCN as ConvOffset2d
from main.model.encoder_v3 import ResCoder, MVCoder

from main.model.flownet import SPyNet
from main.model.inflate import SELayer
from main.utils.utils import Res_Block

# from mmedit.models.losses.perceptual_loss import PerceptualLoss

# perceptual_loss = dict(
#     type='PerceptualLoss',
#     layer_weights={'34': 1.0},
#     vgg_type='vgg19',
#     perceptual_weight=1.0,
#     style_weight=0,
#     norm_img=False)

class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        self.mvCoder = MVCoder(N=128)
        self.resCoder = ResCoder(N=128)
        self.extra_fea = FeaExtra(2)
        self.motion_est = OffsetGen()
        self.mcnet = MCNet(3)
        self.loopfilter = FeatureFix()
        self.mcfilter = LoopFilter()

        self.loss_fn = CharbonnierLoss()

    def forward(self, input_image, refer_frames, enabled_amp, is_compress=False):
        with torch.cuda.amp.autocast(enabled=enabled_amp):
            refer_frame = refer_frames[:, -1, :, :, :].clone()
            input_feat = self.extra_fea(input_image)
            ref_feat = self.extra_fea(refer_frame)
            estmv = self.motion_est(input_feat, ref_feat, input_image, refer_frame)

        with torch.cuda.amp.autocast(enabled=False):
            mv_dict = self.mvCoder.forward(estmv.float())
            mv_aux_loss = self.mvCoder.aux_loss()
            quant_mv_upsample = mv_dict["x_hat"]

            N, _, H, W = input_image.size()
            num_pixels = N * H * W
            bpp_mv = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in mv_dict["likelihoods"].values()
            )

            if is_compress:
                self.mvCoder.eval()
                self.mvCoder.update(force=True)
                out_enc = self.mvCoder.compress(estmv)
                ac_bpp_mv = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

        with torch.cuda.amp.autocast(enabled=enabled_amp):
            prediction1 = self.mcnet(quant_mv_upsample, ref_feat)
            prediction = self.mcfilter(prediction1, refer_frames)
            # 开始图片残差编码
            input_residual = input_feat - prediction

        with torch.cuda.amp.autocast(enabled=False):
            res_dict = self.resCoder.forward(input_residual.float())
            res_aux_loss = self.resCoder.aux_loss()
            recon_res = res_dict["x_hat"]

            N, _, H, W = input_image.size()
            num_pixels = N * H * W
            bpp_res = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in res_dict["likelihoods"].values()
            )

            if is_compress:
                self.resCoder.eval()
                self.resCoder.update(force=True)
                out_enc = self.resCoder.compress(input_residual)
                ac_bpp_res = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

        with torch.cuda.amp.autocast(enabled=enabled_amp):
            recon_image = prediction + recon_res
            recon_image,  recon_fea = self.loopfilter(recon_image, refer_frames)
            recon_image = recon_image.clamp(0., 1.)

        bpp = bpp_res + bpp_mv
        # rloss = torch.nn.L1Loss()(recon_image, input_image)
        rloss = torch.nn.MSELoss()(recon_image, input_image)
        # rloss = self.loss_fn(recon_image, input_image)

        if self.training:
            return recon_image, rloss, bpp, mv_aux_loss, res_aux_loss, recon_fea
        else:
            return recon_image, bpp, recon_fea


class FeaExtra(nn.Module):
    def __init__(self, num_block):
        super(FeaExtra, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.residual_layer = make_layer(Res_Block, num_block)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input):
        out = self.lrelu(self.conv_first(input))
        out = self.residual_layer(out)
        return out


class OffsetGen(nn.Module):
    def __init__(self, num_feat=64):
        super(OffsetGen, self).__init__()
        self.offset_conv11 = nn.ModuleDict()
        self.offset_conv11_1 = nn.ModuleDict()
        self.offset_conv12 = nn.ModuleDict()
        self.feat_fusion = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv11[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            self.offset_conv11_1[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.offset_conv12[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

            if i < 3:
                self.feat_fusion[level] = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.spynet = SPyNet(
            pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth')
        self.attn = SELayer(64)
        self.feat_fusion_ = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

    def forward(self, input_f, ref_f, input_ori, ref_ori):
        x = torch.stack([input_f, ref_f], 1)
        b, t, c, h, w = x.size()
        feat_l2 = self.lrelu(self.conv_l2_1(x.view(-1, c, h, w)))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        feat_l1 = x.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        input_feat = [feat_l1[:, 0, :, :, :].clone(), feat_l2[:, 0, :, :, :].clone(), feat_l3[:, 0, :, :, :].clone()]
        ref_feat = [feat_l1[:, 1, :, :, :].clone(), feat_l2[:, 1, :, :, :].clone(), feat_l3[:, 1, :, :, :].clone()]

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset1 = torch.cat([input_feat[i - 1], ref_feat[i - 1]], dim=1)
            offset1 = self.lrelu(self.offset_conv11[level](offset1))
            offset1 = self.lrelu(self.offset_conv11_1[level](offset1))

            if i == 3:
                offset = self.lrelu(self.offset_conv12[level](offset1))
            else:
                offset = self.lrelu(
                    self.feat_fusion[level](torch.cat([upsampled_offset, offset1], dim=1)))

            if i > 1:
                upsampled_offset = self.upsample(offset)
                upsampled_offset = self.upsample_conv(upsampled_offset)

        flow = self.spynet(input_ori, ref_ori)
        offset = offset + flow.repeat(1, offset.size(1) // 2, 1, 1)

        offset = self.feat_fusion_(offset)
        offset = self.attn(offset)
        return offset


class MCNet(nn.Module):
    def __init__(self, num_block):
        super(MCNet, self).__init__()
        self.dconv = ConvOffset2d(64, 64, 3, stride=1, padding=1, deformable_groups=8)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.recon_layer = make_layer(Res_Block, num_block)
        self.feat_down = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, offset, ref):
        out = self.lrelu(self.dconv(ref, offset))
        out2 = self.conv(torch.cat([out, ref], dim=1))
        out2 = self.lrelu(out2)
        out2 = self.recon_layer(out2)
        return out + out2


class FeatureFix(nn.Module):
    def __init__(self):
        super(FeatureFix, self).__init__()
        self.FeatureExtract_input = FeatureExtract(64, 64, 2)
        self.FeatureExtract_ref = FeatureExtract(3, 64, 2)
        self.recon_layer = make_layer(Res_Block, 2)

        self.conv_10 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv_11 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_12 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv_13 = nn.Conv2d(64, 64, 3, 1, 1)

        self.featfusion = nn.Conv2d(128, 64, 3, 1, 1)
        self.featfusion2 = nn.Conv2d(128, 64, 3, 1, 1)

        self.featdown = nn.Conv2d(64, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.attn = SELayer(64)

        self.stride = 3
        self.ks = 3
        self.pad = self.stride
        self.div = (self.ks / self.stride) ** 2
        self.pool = True

    def forward(self, input_image, refimage):
        N, C, H, W = input_image.size()
        refimage = refimage[:, 0, :, :, :].contiguous().view(-1, 3, H, W)
        inputfeatf = self.FeatureExtract_input(input_image)
        reffeatf = self.FeatureExtract_ref(refimage)

        if self.pool:
            if self.training:
                scale = 8
            else:
                scale = int(inputfeatf.size()[2] / 8)
            inputfeatfp = nn.AvgPool2d(scale, stride=scale)(inputfeatf)
            reffeatfp = nn.AvgPool2d(scale, stride=scale)(reffeatf)
        else:
            inputfeatfp = inputfeatf
            reffeatfp = reffeatf

        inp_patches = F.unfold(inputfeatfp, kernel_size=self.ks, padding=self.pad, stride=self.stride).transpose(2, 1)
        reflr_patches = F.unfold(reffeatfp, kernel_size=self.ks, padding=self.pad, stride=self.stride).transpose(2, 1)
        reflr_patches = reflr_patches.reshape(N, -1, C * self.ks * self.ks)

        _, ind = torch.bmm(F.normalize(inp_patches, dim=2), F.normalize(reflr_patches.transpose(2, 1), dim=1)).max(
            dim=2,
            keepdim=True)  # [N, p*p, Hr*Wr]

        if not self.pool:
            inx = ind.view(N, 1, -1).expand(-1, C * self.ks * self.ks, -1).permute(0, 2, 1)
            c3 = torch.gather(reflr_patches.view(N, reflr_patches.size(1), -1), 1, inx).view(N, -1, C, self.ks, self.ks)
            out_patch = c3.permute(0, 2, 3, 4, 1).view(N, -1, inp_patches.size(1))
            out = F.fold(out_patch, output_size=(H, W), kernel_size=self.ks, padding=self.pad,
                         stride=self.stride) / self.div

            cor = torch.unsqueeze(torch.cosine_similarity(inputfeatf, out), 1)
        else:
            ref_unfold = F.unfold(reffeatf, kernel_size=self.ks * scale, padding=self.pad * scale,
                                  stride=self.stride * scale).transpose(2, 1)
            ref_unfold = ref_unfold.reshape(N, -1, C * self.ks * scale * self.ks * scale)
            index = ind.view(N, 1, -1).expand(-1, C * self.ks * scale * self.ks * scale, -1).permute(0, 2, 1)
            output = torch.gather(ref_unfold, 1, index).view(N, -1, C, self.ks * scale, self.ks * scale)
            output = output.permute(0, 2, 3, 4, 1).view(N, -1, inp_patches.size(1))
            out = F.fold(output, output_size=(H, W), kernel_size=self.ks * scale, padding=self.pad * scale,
                         stride=self.stride * scale) / self.div
            cor = torch.unsqueeze(torch.cosine_similarity(inputfeatf, out), 1)

        out = self.lrelu(self.featfusion(torch.cat([inputfeatf, out], dim=1) * cor))
        # refine
        out = self.lrelu(self.attn(self.featfusion2(torch.cat([out, reffeatf], dim=1))))
        out = self.recon_layer(out)
        input_image = input_image + out
        input_image = self.featdown(input_image)
        return input_image


class LoopFilter(nn.Module):
    def __init__(self):
        super(LoopFilter, self).__init__()
        self.conv01 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv02 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.layer1 = Bottleneck3D()
        self.attn = SELayer(64)
        self.feat_fusion = nn.Conv2d(4 * 64, 64, 1, 1)

    def forward(self, input_image, refer_frames):
        refer_frames = refer_frames[:, 1:, :, :, :]

        N, M, C, H, W = refer_frames.shape
        refer_frames = self.conv01(refer_frames.contiguous().view(N * M, 3, H, W))
        refer_frames = self.conv02(self.lrelu(refer_frames))
        refer_frames = refer_frames.view(N, M, 64, H, W)
        x = torch.cat((refer_frames, input_image.unsqueeze(1)), dim=1)
        x = self.conv1(x.permute(0, 2, 1, 3, 4))
        x = self.lrelu(x)
        x = self.layer1(x)
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b, -1, h, w)
        x = self.feat_fusion(x)
        x = self.lrelu(x)
        x = self.attn(x)
        x = input_image + x
        return x


class Bottleneck3D(nn.Module):
    def __init__(self):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # spatial conv3d kernel
        self.spatial_conv3d = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), \
                                        stride=(1, 1, 1))
        # temporal conv3d kernel
        self.temporal_conv3d = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(0, 0, 0), \
                                         stride=(3, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.spatial_conv3d(out)
        out = out + self.temporal_conv3d(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        out += x
        return out


class FeatureExtract(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(Res_Block, num_blocks, mid_channels)
        self.conv_last = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, x):
        x1 = x = F.leaky_relu(self.conv_first(x))
        x = self.body(x)
        x = self.conv_last(x)
        x = x + x1
        return x


def make_layer(block, num_of_layer, mid_channels=64):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block(mid_channels))
    return nn.Sequential(*layers)
