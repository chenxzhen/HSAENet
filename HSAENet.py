import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np

def SAIs2SAI(x, angRes):
  return rearrange(x, 'b c (u v) h w -> b c (u h) (v w)', u=angRes, v=angRes)


def SAI2SAIs(x, angRes):
  return rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=angRes, v=angRes)

def MacPI2SAI(x, angRes):
  return rearrange(x, 'b c (h u) (w v) -> b c (u h) (v w)', u=angRes, v=angRes)


def SAI2MacPI(x, angRes):
  return rearrange(x, 'b c (u h) (v w) -> b c (h u) (w v)', u=angRes, v=angRes)


class get_model(nn.Module):
    def __init__(self,args):
        super(get_model, self).__init__()
        self.channels = args.channels
        self.angRes = args.angRes_in
        self.factor = args.scale_factor

        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 6
        self.MHSA_params['dropout'] = 0.

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = body(angRes=self.angRes)

        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, lr, info=None):
        # Bicubic
        lr_upscale = interpolate(lr, self.angRes, scale_factor=self.factor, mode='bicubic')
        # [B(atch), 1, A(ngRes)*h(eight)*S(cale), A(ngRes)*w(idth)*S(cale)]

        # reshape for LFT
        lr = rearrange(lr, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        # [B, C(hannels), A^2, h, w]

        # Initial Convolution
        buffer = self.conv_init0(lr)
        buffer = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]

        # Alternate AngTrans & SpaTrans
        buffer = self.altblock(buffer) + buffer

        # Up-Sampling
        buffer = rearrange(buffer, 'b c (a1 a2) h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        buffer = self.upsampling(buffer)
        out = buffer + lr_upscale

        return out
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        B = x.size()[0]
        x = self.avgpool(x)
        x = self.fc(x)
        if B == 1:
            pass
        else:
            x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        # x = SAIs2SAI(x,5)
        x = self._forward_impl(x)
        # x = SAI2SAIs(x,5)
        return x

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )

        self.local_att_epi = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )
        self.local_att2_epi = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )
        self.global_att_epi = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )

        self.global_att2_epi = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )


        self.local_att_mac = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )
        self.local_att2_mac = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )
        self.global_att_mac = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )

        self.global_att2_mac = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x,y,z, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xb = y + residual
        xle = self.local_att_epi(xb)
        xge = self.global_att_epi(xb)
        xlge = xle + xge
        weie = self.sigmoid(xlge)

        xc = z + residual
        xlm = self.local_att_mac(xc)
        xgm = self.global_att_mac(xc)
        xlgm = xlm + xgm
        weim = self.sigmoid(xlgm)

        xi = (x * wei + residual * (1 - wei) + y * weie + residual * (1 - weie) + z * weim + residual * (1 - weim))/3

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)

        xle2 = self.local_att2_epi(xi)
        xge2 = self.global_att2_epi(xi)
        xlge2 = xle2 + xge2
        weie2 = self.sigmoid(xlge2)

        xlm2 = self.local_att2_mac(xi)
        xgm2 = self.global_att2_mac(xi)
        xlgm2 = xlm2 + xgm2
        weim2 = self.sigmoid(xlgm2)

        xo = (x * wei2 + residual * (1 - wei2) + y * weie2 + residual * (1 - weie2) + z * weim2 + residual * (1 - weim2))/3

        return xo

class MacPI(nn.Module):
    def __init__(self,angRes):
        super(MacPI,self).__init__()
        self.angRes = angRes
        # self.macPI_ang1 = nn.Sequential(
        #     ODConv2d(60, 25, kernel_size=1, dilation=1, padding=0, stride=1, reduction=0.0625, kernel_num=4),
        #     ODConv2d(25, 25, kernel_size=5, dilation=1, padding=2, stride=5, groups=5, reduction=0.0625, kernel_num=4),
        #     nn.ReLU()
        # )

        self.macPI_ang1 = CSWinIR(is_spa=False,is_macPI=True,split_size=5, transtype=5, angRes=angRes, depths=1)
        self.macPI_ang2 = CSWinIR(is_spa=False,is_macPI=True,split_size=10, transtype=5, angRes=angRes, depths=1)
        # self.macPI_ang2 = nn.Sequential(
        #     ODConv2d(60, 25, kernel_size=1, dilation=1, padding=0, stride=1, reduction=0.0625, kernel_num=4),
        #     ODConv2d(25, 25, kernel_size=5 * 2, dilation=1, padding=3, stride=5, groups=5, reduction=0.0625,
        #              kernel_num=4),
        #     nn.ReLU()
        # )
        # self.conv1x1_1 = nn.Conv3d(1,60,1)
        # self.conv1x1_2 = nn.Conv3d(1, 60, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b,c,a,h,w = x.size()
        # y_copy = rearrange(x, 'b c (a1 a2) h w -> b c (h a1) (w a2)', a1=self.angRes, a2=self.angRes)

        y = self.macPI_ang1(x)
        y2 = self.macPI_ang2(x)
        # y = self.conv1x1_1(y)
        # y2 = self.conv1x1_1(y2)
        y = y + y2
        y = self.sigmoid(y)
        y = x * y.expand_as(x)+x
        return y


class AltFilter(nn.Module):
    def __init__(self, angRes):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.spa_trans = CSWinIR(is_spa=True, transtype=1, angRes=angRes, depths=2)
        self.ang_trans = CSWinIR(is_spa=False, transtype=2, angRes=angRes, depths=2)

        self.softmax_ = nn.Softmax(dim=2)
        # self.conv_de = nn.Conv3d(60*3, 60, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False)
        self.macPI = MacPI(angRes)
        self.EPIFilter = EPIFilter(angRes=angRes)
        self.iAFF = iAFF(60, 6)
    def forward(self, buffer):
        buffer1 = self.spa_trans(self.ang_trans(buffer))
        buffer2 = self.EPIFilter(buffer)
        buffer3 = self.macPI(buffer)
        # buffer_tmp = torch.cat([buffer1,buffer2,buffer3],dim = 1)
        buffer11 = self.iAFF(buffer1,buffer2,buffer3,buffer)
        return buffer11

class EPIFilter(nn.Module):
    def __init__(self, angRes):
        super(EPIFilter, self).__init__()

        self.epiw_trans = CSWinIR(is_spa=False, transtype=3, angRes=angRes, depths=1)
        self.epih_trans = CSWinIR(is_spa=False, transtype=4, angRes=angRes, depths=1)

    def forward(self, buffer):
        buffer = self.epiw_trans(buffer)
        buffer = self.epih_trans(buffer)
        return buffer

class body(nn.Module):
    def __init__(self, angRes):
        super(body, self).__init__()
        self.AltFilter1 = AltFilter(angRes=angRes)
        self.AltFilter2 = AltFilter(angRes=angRes)
        self.AltFilter3 = AltFilter(angRes=angRes)
        self.AltFilter4 = AltFilter(angRes=angRes)
        self.AltFilter5 = AltFilter(angRes=angRes)
        self.AltFilter6 = AltFilter(angRes=angRes)
        self.AltFilter7 = AltFilter(angRes=angRes)
        self.AltFilter8 = AltFilter(angRes=angRes)
        # self.EPIFilter = EPIFilter(angRes=angRes)

    def forward(self, buffer):

        buffer1 = self.AltFilter1(buffer)
        buffer2 = self.AltFilter2(buffer1)
        buffer3 = self.AltFilter3(buffer2)
        buffer4 = self.AltFilter4(buffer3)
        buffer5 = self.AltFilter5(buffer4)
        buffer6 = self.AltFilter6(buffer5)
        buffer7 = self.AltFilter7(buffer6)
        buffer8 = self.AltFilter8(buffer7)

        # buffer9 = self.EPIFilter(buffer8)
        return buffer8

def interpolate(x, angRes, scale_factor, mode):
    [B, _, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, 1, angRes, h, angRes, w)
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor,
                                                                      W * scale_factor)  # [B, 1, A*h*S, A*w*S]
    return x_upscale


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, info=None):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):
    pass


class LeFF(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Sequential(nn.Linear(in_features, hidden_features),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, groups=hidden_features, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_features, out_features))
        self.dim = in_features
        self.hidden_dim = hidden_features

    def forward(self, x, x_size):
        # bs x hw x c
        B, L, C = x.shape
        H, W = x_size
        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)

        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, idy=None, split_size=7, dim_out=None, num_heads=8, attn_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == None:
            pass
        else:
            if idx == -1:  # global attenton
                H_sp, W_sp = self.resolution[0], self.resolution[1]
            elif idx == 0:  # row attention
                H_sp, W_sp = self.resolution[0], self.split_size
            elif idx == 1:  # column attention
                W_sp, H_sp = self.resolution[1], self.split_size
            else:
                print("ERROR MODE", idx)
                exit(0)
        if idy == None:
            pass
        else:
            if idy == 0:  # column attention
                W_sp, H_sp = self.split_size, self.split_size
            elif idy == 1:  # column attention
                W_sp, H_sp = 1, self.split_size
            elif idy == 2:  # column attention
                W_sp, H_sp = self.split_size,1

        self.H_sp = H_sp
        self.W_sp = W_sp
        # self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        if not self.H_sp == self.split_size:
            self.H_sp = H
        if not self.W_sp == self.split_size:
            self.W_sp = W
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, x_size, func):
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, x_size, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H, W = x_size
        B, L, C = q.shape

        q = self.im2cswin(q, x_size)
        k = self.im2cswin(k, x_size)
        v = self.im2cswin(v, x_size)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, is_macPI=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        if is_macPI:
            self.branch_num = 3
            self.attns = nn.ModuleList([
                # LePEAttention(
                #     dim // self.branch_num, resolution=self.patches_resolution,idx=None, idy=i,
                #     split_size=split_size, num_heads=num_heads // self.branch_num, dim_out=dim // self.branch_num,
                #     qk_scale=qk_scale, attn_drop=attn_drop)
                # for i in range(self.branch_num)])
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=None, idy=i,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for i in range(1)])
        else:
            if last_stage:
                self.attns = nn.ModuleList([
                    LePEAttention(
                        dim, resolution=self.patches_resolution, idx=-1,
                        split_size=split_size, num_heads=num_heads, dim_out=dim,
                        qk_scale=qk_scale, attn_drop=attn_drop)
                    for i in range(self.branch_num)])
            else:
                self.attns = nn.ModuleList([
                    LePEAttention(
                        dim // 2, resolution=self.patches_resolution, idx=i,idy=None,
                        split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                        qk_scale=qk_scale, attn_drop=attn_drop)
                    for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = LeFF(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape

        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](x_size, qkv[:, :, :, :C // 2])
            x2 = self.attns[1](x_size, qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        elif self.branch_num == 3:
            x1 = self.attns[0](x_size, qkv)
            # x2 = self.attns[1](x_size, qkv[:, :, :, C // 3: (C // 3) * 2])
            # x3 = self.attns[2](x_size, qkv[:, :, :, (C // 3) * 2:])
            # attened_x = torch.cat([x1, x2, x3], dim=2)
            attened_x = x1
        else:
            attened_x = self.attns[0](qkv)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class PA(nn.Module):
    def __init__(self, dim, x_size, kernel_size):
        super().__init__()
        if kernel_size == 3:
            self.pa_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, groups=dim)
        elif kernel_size == 5:
            self.pa_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=2, groups=dim)

        self.sigmoid = nn.Sigmoid()
        self.x_size = x_size
        self.dim = dim

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, with_pos=True,
                 is_spa=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.with_pos = with_pos

        if is_spa:
            self.pos = PA(embed_dim, patches_resolution, kernel_size=3)
        else:
            self.pos = PA(embed_dim, patches_resolution, kernel_size=5)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, split_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 img_size=224, patch_size=4, is_spa=True, is_macPI=False):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.is_spa = is_spa

        self.residual_group = nn.ModuleList([
            CSWinBlock(dim=dim, num_heads=num_heads, reso=input_resolution, mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[i], norm_layer=norm_layer, is_macPI=is_macPI)
            for i in range(depth)])

        if is_spa:
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None, is_spa=is_spa)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        x1 = x
        x = self.patch_embed(x)
        for layer in self.residual_group:
            x = layer(x, x_size)
        if self.is_spa:
            x = self.conv(self.patch_unembed(x, x_size)) + x1
        else:
            x = self.patch_unembed(x, x_size)
        return x


class CSWinIR(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=64, patch_size=1, embed_dim=60, depths=2,
                 split_size=1,
                 num_heads=6, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_chk=False, patch_norm=True, is_spa=True,
                 angRes=5, transtype=1, is_macPI=False):
        super().__init__()

        self.split_size = split_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.is_spa = is_spa
        self.angRes = angRes
        self.transtype = transtype
        self.is_macPI = is_macPI

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.use_chk = use_chk

        self.pos_drop = nn.Dropout(p=drop_rate)

        y = torch.linspace(0, drop_path_rate, depths)
        dpr = [x.item() for x in y]  # stochastic depth decay rule

        self.layer = RSTB(dim=embed_dim,
                          input_resolution=(patches_resolution[0],
                                            patches_resolution[1]),
                          depth=depths,
                          num_heads=num_heads,
                          split_size=split_size,
                          mlp_ratio=self.mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path=dpr,  # no impact on SR results
                          norm_layer=norm_layer,
                          img_size=img_size,
                          patch_size=patch_size,
                          is_spa=self.is_spa,
                          is_macPI=self.is_macPI
                          )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def check_image_size(self, x, transtype):
        b, c, a, h, w = x.size()

        if transtype == 1:
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * a, c, h, w)
            mod_pad_h = (self.split_size - h % self.split_size) % self.split_size
            mod_pad_w = (self.split_size - w % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        elif transtype == 2:
            x = rearrange(x, 'b c (a1 a2) h w -> (b h w) c a1 a2', a1=self.angRes, a2=self.angRes)
            mod_pad_a = (self.split_size - self.angRes % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_a, 0, mod_pad_a), 'reflect')
        elif transtype == 3:
            x = rearrange(x, 'b c (a1 a2) h w -> (b h a1) c w a2', a1=self.angRes, a2=self.angRes)
            mod_pad_a = (self.split_size - self.angRes % self.split_size) % self.split_size
            mod_pad_w = (self.split_size - w % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_a, 0, mod_pad_w), 'reflect')
        elif transtype == 4:
            x = rearrange(x, 'b c (a1 a2) h w -> (b w a2) c h a1', a1=self.angRes, a2=self.angRes)
            mod_pad_a = (self.split_size - self.angRes % self.split_size) % self.split_size
            mod_pad_h = (self.split_size - h % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_a, 0, mod_pad_h), 'reflect')
        elif transtype == 5:
            x = rearrange(x, 'b c (a1 a2) h w -> b c (h a1) (w a2)', a1=self.angRes, a2=self.angRes)
            mod_pad_h = (self.split_size - (h * self.angRes) % self.split_size) % self.split_size
            mod_pad_w = (self.split_size - (w * self.angRes) % self.split_size) % self.split_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.pos_drop(x)
        x = self.layer(x, x_size)
        return x

    def forward(self, x):
        b, c, a, H, W = x.shape
        x = self.check_image_size(x, self.transtype)
        x = self.forward_features(x)
        if self.transtype == 2:
            x = rearrange(x, '(b h w) c a1 a2->(b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=H, w=W)
        if self.transtype == 3:
            x = rearrange(x, '(b h a1) c w a2->(b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=H, w=W)
        if self.transtype == 4:
            x = rearrange(x, '(b w a2) c h a1->(b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=H, w=W)
        if self.transtype == 5:
            x = rearrange(x, 'b c (h a1) (w a2)->(b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=H, w=W)
        x = x[:, :, :H, :W]
        x = x.view(b, a, c, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return x


if __name__ == "__main__":
    from option import args
    net = get_model(args).cuda()
    from thop import profile

    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))

#    Number of parameters: 1.57M
#    Number of FLOPs: 39.05G
