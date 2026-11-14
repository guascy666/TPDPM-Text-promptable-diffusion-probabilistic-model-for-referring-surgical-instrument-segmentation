import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm


from accelerate import Accelerator



#from version import *
import sys
sys.path.append('/media/cygzz/data/guanjl/Code_self/models')
from attend import *
from basic_model import *
from resunet import *
from Image_Text_Fusion import *
import torch
import torch.nn as nn
import numpy as np
from text_module import *
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model
class Diff(Module):
    def __init__(self):
        super().__init__()
        self.backbone=Unet()
        self.fusion=Condition_Encoder(channel=3)
        self.text_encoder=Text_projector()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
    
    def forward(self,x,t,img,text):
        text_emb=self.text_encoder(text)
        time_emb=self.time_mlp(t)
        fusion_emb=self.fusion(img,text_emb,time_emb)
        out=self.backbone(x,t,img,fusion_emb)
        return out

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 计算交集
        intersection = (inputs * targets).sum(dim=(1,2,3))
        # 计算并集
        union = inputs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()
        # 计算 Dice Loss
        dice_loss = 1 - dice
        return dice_loss

class Diff2(Module):
    def __init__(self):
        super().__init__()
        self.backbone=Unet(out_dim=2)
        self.fusion=Condition_Encoder(channel=3)
        self.text_encoder=Text_projector()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
    
    def forward(self,x,t,img,text):
        text_emb=self.text_encoder(text)
        time_emb=self.time_mlp(t)
        fusion_emb=self.fusion(img,text_emb,time_emb)
        out=self.backbone(x,t,img,fusion_emb)
        return out


class Diff3(Module):
    def __init__(self):
        super().__init__()
        self.backbone=Unet()
        self.fusion=Condition_Encoder(channel=3)
        self.text_encoder=Text_projector()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
    
    def forward(self,x,t,img,text):
        text_emb=self.text_encoder(text)
        time_emb=self.time_mlp(t)
        fusion_emb=self.fusion(img,text_emb,time_emb)
        out=self.backbone(x,t,img,fusion_emb)
        return out
    
class Diff4(Module):
    def __init__(self):
        super().__init__()
        self.backbone=Unet3(out_dim=1)
        self.fusion=Condition_Encoder(channel=3)
        self.img_encoder=con_coder()
        self.text_encoder=Text_projector()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
    
    def forward(self,x,t,img,text):
        cc_list=self.img_encoder(img)
        text_emb=self.text_encoder(text)
        time_emb=self.time_mlp(t)
        fusion_emb=self.fusion(img,text_emb,time_emb)
        out=self.backbone(x,t,img,fusion_emb,cc_list,text_emb)
        return out

class Diff5(Module):
    def __init__(self):
        super().__init__()
        self.backbone=Unet3(out_dim=1)
        self.fusion=Condition_Encoderv2(channel=3)
        self.img_encoder=con_coder()
        self.text_encoder=Text_projector()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
    
    def forward(self,x,t,img,text):
        cc_list=self.img_encoder(img)
        text_emb=self.text_encoder(text)
        time_emb=self.time_mlp(t)
        fusion_emb=self.fusion(cc_list[-1],text_emb,time_emb)
        out=self.backbone(x,t,img,fusion_emb,cc_list,text_emb)
        return out

class Condition_Encoderv2(nn.Module):
    def __init__(self, channel=3,d_model=512,filters=[64, 128, 256, 512]):
        super(Condition_Encoderv2, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        #self.fusion1=Encoder(64,d_model)
        self.fusion2=Encoder(128,d_model)
        self.fusion3=Encoder(256,d_model)
        self.fusion4=Encoder(512,d_model)
    
    def forward(self, img, text_emb=None, time_emb=None):
        # Encode
        # x1 = self.input_layer(img) + self.input_skip(img)
        
        # x2 = self.residual_conv_1(x1,time_emb)
        # x2 = self.fusion2(x2,text_emb)
        # x3 = self.residual_conv_2(x2,time_emb)
        # x3 = self.fusion3(x3,text_emb)
        # x4 = self.residual_conv_3(x3,time_emb)
        fusion_out=self.fusion4(img,text_emb)

        return fusion_out
    
class Unet2(Module):
    def __init__(
        self,
        dim=64,
        init_dim = 32,
        out_dim = 1,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        self_condition = True,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = 6

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.Fusion=MIDTransformer(512)

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None,fusion_emb=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        # if self.self_condition:
        #     x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.Fusion(x,fusion_emb)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
class Unet(Module):
    def __init__(
        self,
        dim=64,
        init_dim = 32,
        out_dim = 1,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        self_condition = True,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = 4

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.Fusion=MIDTransformer(512)

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None,fusion_emb=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        # if self.self_condition:
        #     x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.Fusion(x,fusion_emb)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
from Image_Text_Fusion import Encoder
class Unet3(Module):
    def __init__(
        self,
        dim=64,
        init_dim = 32,
        out_dim = 1,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        self_condition = True,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = 4

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.Fusion=MIDTransformer(512)

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        self.fu = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)
            #is_last=False

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in+dim_out, dim_out) if not is_last else nn.Conv2d(dim_in+dim_out, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
            self.fu.append(Encoder(channel=dim_out))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None,fusion_emb=None,cc_list=None,text_feature=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        # if self.self_condition:
        #     x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        i=0

        for block1, block2, attn, downsample in self.downs:
            item=cc_list[i]
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x=torch.cat([x,item],dim=1)

            x = downsample(x)
            i+=1

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.Fusion(x,fusion_emb)
        x = self.mid_block2(x, t)
        
        j=0
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x
            x=self.fu[j](x,text_feature)
            j+=1
            

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        image_size,
        timesteps = 500,
        sampling_timesteps = 10,#1,(50) 2(20) 3(10)
        objective = 'pred_x0',
        beta_schedule = 'linear',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        ##assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = 1
        #self.self_condition = self.model.self_condition

        # if isinstance(image_size, int):
        #     image_size = (image_size, image_size)
        #assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond_img,text, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, cond_img,text)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, cond_img,text,x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, cond_img,text)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int,cond_img,text, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond_img=cond_img,text=text,clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape,cond_img,text,return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

       
        return ret

    @torch.inference_mode()
    def ddim_sample_diff(self, shape, cond_img,text,return_all_timesteps = False):
        print("ddim ing!!")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # img = torch.randn(shape, device = device)
        # imgs = [img]

        # x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        #     time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        #     #self_cond = x_start if self.self_condition else None
        #     pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img,text, clip_x_start = True, rederive_pred_noise = True)

        #     if time_next < 0:
        #         img = x_start
        #         imgs.append(img)
        #         continue

        #     alpha = self.alphas_cumprod[time]
        #     alpha_next = self.alphas_cumprod[time_next]

        #     sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        #     c = (1 - alpha_next - sigma ** 2).sqrt()

        #     noise = torch.randn_like(img)
        #     #x_start=2*torch.sigmoid(x_start)-1
        #     #x_start=2*x_start-1
        #     img = x_start * alpha_next.sqrt() + \
        #           c * pred_noise + \
        #           sigma * noise

        #     imgs.append(img)

        # ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # #ret = self.unnormalize(ret)
        # #ret = torch.sigmoid(ret)
        # return ret
        ret_set=[]
        diff_jet=[]
        for i in range(10):
            img = torch.randn(shape, device = device)
            img=img.cuda()
            imgs = [img]

            x_start = None

            for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                #self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img,text, clip_x_start = True, rederive_pred_noise = True)

                if time_next < 0:
                    img = x_start
                    imgs.append(img)
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                #x_start=torch.sigmoid(x_start)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

                imgs.append(img)

            ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
            ret_set.append(ret)
            diff_jet.append(imgs)

        #ret = self.unnormalize(ret)
        #ret = torch.sigmoid(ret)
        return imgs
    
    def ddim_sample_once(self, shape, cond_img,text,return_all_timesteps = False):
        print("ddim ing!!")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # img = torch.randn(shape, device = device)
        # imgs = [img]

        # x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        #     time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        #     #self_cond = x_start if self.self_condition else None
        #     pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img,text, clip_x_start = True, rederive_pred_noise = True)

        #     if time_next < 0:
        #         img = x_start
        #         imgs.append(img)
        #         continue

        #     alpha = self.alphas_cumprod[time]
        #     alpha_next = self.alphas_cumprod[time_next]

        #     sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        #     c = (1 - alpha_next - sigma ** 2).sqrt()

        #     noise = torch.randn_like(img)
        #     #x_start=2*torch.sigmoid(x_start)-1
        #     #x_start=2*x_start-1
        #     img = x_start * alpha_next.sqrt() + \
        #           c * pred_noise + \
        #           sigma * noise

        #     imgs.append(img)

        # ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # #ret = self.unnormalize(ret)
        # #ret = torch.sigmoid(ret)
        # return ret
        ret_set=[]
        for i in range(10):
            img = torch.randn(shape, device = device)
            img=img.cuda()
            imgs = [img]

            x_start = None

            for time, time_next in time_pairs:
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                #self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img,text, clip_x_start = True, rederive_pred_noise = True)

                if time_next < 0:
                    img = x_start
                    imgs.append(img)
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                #x_start=torch.sigmoid(x_start)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

                imgs.append(img)

            ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
            ret_set.append(ret)

        #ret = self.unnormalize(ret)
        #ret = torch.sigmoid(ret)
        return ret_set
    
    def sample_once(self,cond_img,text,batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample_once
        return sample_fn((cond_img.shape[0],channels, h, w), cond_img,text,return_all_timesteps = return_all_timesteps)
    
    def ddim_sample(self, shape, cond_img,text,return_all_timesteps = False):
        #print("ddim ing!!")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # img = torch.randn(shape, device = device)
        # imgs = [img]

        # x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        #     time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        #     #self_cond = x_start if self.self_condition else None
        #     pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img,text, clip_x_start = True, rederive_pred_noise = True)

        #     if time_next < 0:
        #         img = x_start
        #         imgs.append(img)
        #         continue

        #     alpha = self.alphas_cumprod[time]
        #     alpha_next = self.alphas_cumprod[time_next]

        #     sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        #     c = (1 - alpha_next - sigma ** 2).sqrt()

        #     noise = torch.randn_like(img)
        #     #x_start=2*torch.sigmoid(x_start)-1
        #     #x_start=2*x_start-1
        #     img = x_start * alpha_next.sqrt() + \
        #           c * pred_noise + \
        #           sigma * noise

        #     imgs.append(img)

        # ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # #ret = self.unnormalize(ret)
        # #ret = torch.sigmoid(ret)
        # return ret
        ret_set=[]
        for i in range(10):
            img = torch.randn(shape, device = device)
            img=img.cuda()
            imgs = [img]

            x_start = None

            for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                #self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img,text, clip_x_start = True, rederive_pred_noise = True)

                if time_next < 0:
                    img = x_start
                    imgs.append(img)
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                #x_start=torch.sigmoid(x_start)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

                imgs.append(img)

            ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
            ret_set.append(ret)

        #ret = self.unnormalize(ret)
        #ret = torch.sigmoid(ret)
        return ret_set
    
    def ddim_cla(self,shape,cond_img,text,classifier,return_all_timesteps=False):
        print("ddim ing!!")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # img = torch.randn(shape, device = device)
        # imgs = [img]

        # x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        #     time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        #     #self_cond = x_start if self.self_condition else None
        #     pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img,text, clip_x_start = True, rederive_pred_noise = True)

        #     if time_next < 0:
        #         img = x_start
        #         imgs.append(img)
        #         continue

        #     alpha = self.alphas_cumprod[time]
        #     alpha_next = self.alphas_cumprod[time_next]

        #     sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        #     c = (1 - alpha_next - sigma ** 2).sqrt()

        #     noise = torch.randn_like(img)
        #     #x_start=2*torch.sigmoid(x_start)-1
        #     #x_start=2*x_start-1
        #     img = x_start * alpha_next.sqrt() + \
        #           c * pred_noise + \
        #           sigma * noise

        #     imgs.append(img)

        # ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # #ret = self.unnormalize(ret)
        # #ret = torch.sigmoid(ret)
        # return ret
        ret_set=[]
        #range(10)
        for i in range(10):
            img = torch.randn(shape, device = device)
            imgs = [img]

            x_start = None

            for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                #self_cond = x_start if self.self_condition else None
                pred_noise, x_start= self.model_predictions(img, time_cond, cond_img,text, clip_x_start = True, rederive_pred_noise = True)

                if time_next < 0:
                    img = x_start
                    imgs.append(img)
                    continue
                with torch.enable_grad():
                    x_start.requires_grad = True
                    cond_img.requires_grad = True
                    logits = classifier(torch.sigmoid(x_start),cond_img)
                    log_probs = F.log_softmax(logits, dim=-1)
                    # 假设目标类别为0
                    log_probs=log_probs.squeeze(0)
                    target = torch.zeros(log_probs.shape[0], dtype=torch.long).cuda()
                    loss = F.nll_loss(log_probs, target)
                    grad = torch.autograd.grad(loss, x_start)[0]
                x0_pred_guided = x_start + 1* grad
                time_tensor=torch.tensor(time).cuda()
                time_tensor=time_tensor.long()
                time_tensor=time_tensor.view(-1)
                pred_noise_guided = self.predict_noise_from_start(img, time_tensor, x0_pred_guided)


                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                #x_start=torch.sigmoid(x_start)
                img = x0_pred_guided * alpha_next.sqrt() + \
                    c * pred_noise_guided + \
                    sigma * noise

                imgs.append(img)

            ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
            ret_set.append(ret)

        #ret = self.unnormalize(ret)
        #ret = torch.sigmoid(ret)
        return ret_set


    @torch.inference_mode()
    def sample_diff(self,cond_img,text, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample_diff
        return sample_fn((cond_img.shape[0],channels, h, w), cond_img,text,return_all_timesteps = return_all_timesteps)
    
    def sample(self,cond_img,text, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((cond_img.shape[0],channels, h, w), cond_img,text,return_all_timesteps = return_all_timesteps)
    
    def sample_cla(self,cond_img,text,cla, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_cla
        return sample_fn((cond_img.shape[0],channels, h, w), cond_img,text,cla,return_all_timesteps = return_all_timesteps)
    
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
   

    def p_losses(self, x_start, t,cond_img,text, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        noise=noise.cuda()

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, cond_img,text)
        

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
            #target = self.unnormalize(target)
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        #########################v3
        # model_out=torch.sigmoid(model_out)
        # loss2=DiceLoss()
        # loss=loss2(model_out,target)
        ################################
        model_out=torch.sigmoid(model_out)
        loss2=DiceLoss()
        target=self.unnormalize(target)
        loss=loss2(model_out,target)
        return loss,model_out

    def forward(self, img, cond_img,text,*args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        #assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        t=t.cuda()
        #########v1
        img = self.normalize(img)
        # img.cuda()
        ############v2
        return self.p_losses(img, t, cond_img,text,*args, **kwargs)

