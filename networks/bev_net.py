import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from networks.vrwkv import Block as RWKVBlock

class BEVFusionv1(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.attention_bev = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_sem = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_com = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.adapter_sem = nn.Conv2d(channel//2, channel, 1)
        self.adapter_com = nn.Conv2d(channel//2, channel, 1)

    def forward(self, bev_features, sem_features, com_features):
        sem_features = self.adapter_sem(sem_features)
        com_features = self.adapter_com(com_features
        )
        attn_bev = self.attention_bev(bev_features)
        attn_sem = self.attention_sem(sem_features)
        attn_com = self.attention_com(com_features)

        fusion_features = torch.mul(bev_features, attn_bev) \
            + torch.mul(sem_features, attn_sem) \
            + torch.mul(com_features, attn_com)

        return fusion_features


class BEVUNetv1(nn.Module):
    def __init__(self, n_class, n_height, dilation, bilinear, group_conv, input_batch_norm, dropout, circular_padding, dropblock):
        super().__init__()
        # Encoder
        self.inc = inconv(64, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        
        # RWKV blocks for the encoder
      
        self.rwkv_block1 = RWKVBlock(n_embd=128, n_layer=18, layer_id=0)
        self.rwkv_block2 = RWKVBlock(n_embd=256, n_layer=18, layer_id=0)
        self.rwkv_block3 = RWKVBlock(n_embd=512, n_layer=18, layer_id=0)
        self.rwkv_block4 = RWKVBlock(n_embd=512, n_layer=18, layer_id=0)
        self.rwkv_block_up1 = RWKVBlock(n_embd=512, n_layer=18, layer_id=0)
        self.rwkv_block_up2 = RWKVBlock(n_embd=256, n_layer=18, layer_id=0)
        self.rwkv_block_up3 = RWKVBlock(n_embd=128, n_layer=18, layer_id=0)
        self.rwkv_block_up4 = RWKVBlock(n_embd=128, n_layer=18, layer_id=0)

        # BEV fusion modules
        channels = [128, 256, 512]
        self.bev_fusions = nn.ModuleList([BEVFusionv1(channels[i]) for i in range(3)])

        # Decoder
        self.up1 = up(1024, 512, circular_padding, bilinear=bilinear, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up2 = up(768, 256, circular_padding, bilinear=bilinear, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up3 = up(384, 128, circular_padding, bilinear=bilinear, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up4 = up(192, 128, circular_padding, bilinear=bilinear, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.outc = outconv(128, n_class)
        
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)

    def forward(self, x, sem_fea_list, com_fea_list):
        # Encoder with RWKV blocks
        x1 = self.inc(x)
       
        x2 = self.down1(x1)
        B, C, H, W = x2.shape
        x2 = self.rwkv_block1(x2.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=(H, W)).view(B, H, W, C).permute(0, 3, 1, 2)
        x2_f = self.bev_fusions[0](x2, sem_fea_list[0], com_fea_list[0])

        x3 = self.down2(x2_f)
        B, C, H, W = x3.shape
        x3 = self.rwkv_block2(x3.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=(H, W)).view(B, H, W, C).permute(0, 3, 1, 2)
        x3_f = self.bev_fusions[1](x3, sem_fea_list[1], com_fea_list[1])

        x4 = self.down3(x3_f)
        B, C, H, W = x4.shape
        x4 = self.rwkv_block3(x4.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=(H, W)).view(B, H, W, C).permute(0, 3, 1, 2)
        x4_f = self.bev_fusions[2](x4, sem_fea_list[2], com_fea_list[2])
        
        x5 = self.down4(x4_f)
        B, C, H, W = x5.shape
        x5 = self.rwkv_block4(x5.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=(H, W)).view(B, H, W, C).permute(0, 3, 1, 2)
        
       
        # return x
        x = self.up1(x5, x4_f)  # 512, 512
        x = self.up2(x, x3_f)  # 512, 256
        x = self.up3(x, x2_f)  # 256, 128
        x = self.up4(x, x1)  # 128, 64
        x = self.outc(self.dropout(x))
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        #add circular padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False, use_dropblock=False, drop_p=0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, groups = in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch,group_conv = group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch, group_conv = group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
