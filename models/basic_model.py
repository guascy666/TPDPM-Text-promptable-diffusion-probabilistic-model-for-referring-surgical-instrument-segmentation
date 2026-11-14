import torch.nn as nn
import torch
from torch import einsum
import math
from einops import rearrange
from Image_Text_Fusion import Encoder
def exists(x):
    return x is not None

def default(val, d):
    # 判断变量是否存在, 如果存在, 直接返回结果. 否则进行变量2的判断
    if exists(val):
        return val

    # 判断输入变量2是否为函数, 如果为函数, 返回函数结果, 否则直接返回变量2
    if callable(d):
        return d()
    else:
        return d


# 残差链接，直接返回结果
def identity(t):
    return t


# 归一化函数, 从 [0, 1] 放缩至 [-1, 1]
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


# 归一化函数, 从 [-1, 1] 放缩至 [0, 1]
def un_normalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding,time_emb_dim=512):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        self.mlp=nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, output_dim * 2)) 
        self.act=nn.SiLU()

    def forward(self, x,time_emb=None,ifi=False):
        scale_shift = None
        if ifi:
        #if exists(self.mlp) and ((time_emb==None)==False):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
            scale,shift=scale_shift
            y = self.conv_block(x)
            y = y*(scale+1) + shift
            y=self.act(y)

            return y+self.conv_skip(x)
        else:
            return self.act(self.conv_block(x))+self.conv_skip(x)
        
        #return self.conv_block(x)+self.conv_skip(x) 


class ResidualConv_Fusion(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding,time_emb_dim=512):
        super(ResidualConv_Fusion, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        self.mlp=nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, output_dim * 2)) 
        self.act=nn.SiLU()
        self.fusion=Encoder(channel=output_dim)

    def forward(self, x,text_emb=None,time_emb=None):
        scale_shift = None
        
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale_shift = time_emb.chunk(2, dim=1)
        scale,shift=scale_shift
        y = self.conv_block(x)
        y = y*(scale+1) + shift
        y=self.act(y)
        y=self.fusion(y,text_emb)

        return y+self.conv_skip(x)
        
        #return self.conv_block(x)+self.conv_skip(x)

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)
    

# 正弦位置编码模块
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MIDAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        # 超参数设置(注意力头数目 归一化因子 注意力隐藏层维度)
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        # 设置归一化层、QKV转换层、注意力输出层
        self.pre_norm_x = LayerNorm(dim)
        self.pre_norm_c = LayerNorm(dim)

        self.to_qkv_x = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_qkv_c = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, (1, 1))

    def forward(self, x, c_x):
        # 获取数据维度
        b, c, h, w = x.shape

        # 数据归一化后划分为 Q K V 注意力机制
        x = self.pre_norm_x(x)
        c_x = self.pre_norm_c(c_x)

        qkv_x = self.to_qkv_x(x).chunk(3, dim=1)
        qkv_c = self.to_qkv_c(c_x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_x)
        q_c, k_c, v_c = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_c)

        # 获取 QKV 的计算结果
        q_c = q_c * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q_c, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # 维度转换后获取网络输出
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

def feed_forward_att(dim, mult=4):
    inner_dim = int(dim * mult)
    feed_forward_linear = nn.Sequential(LayerNorm(dim),
                                        nn.Conv2d(dim, inner_dim, (1, 1)),
                                        nn.GELU(),
                                        nn.Conv2d(inner_dim, dim, (1, 1)))
    return feed_forward_linear

class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class MIDTransformer(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(MIDAttention(dim, dim_head=dim_head, heads=heads)),
                Residual(feed_forward_att(dim)),
                Residual(feed_forward_att(dim)),
                Residual(MIDAttention(dim, dim_head=dim_head, heads=heads))]))

    def forward(self, x, c):
        for attn_1, ff_linear_1, ff_linear_2, attn_2 in self.layers:
            x = attn_1(x, c)
            x1 = ff_linear_1(x)
            x2 = ff_linear_2(x)
            x = attn_2(x1, x2)
        return x

if __name__=="__main__":
    x=torch.zeros((1,3,128,128))
    fusion_emb=torch.zeros((1,512,28,28))
    time=torch.zeros((1,512))
    
    x=x.cuda()
    fusion_emb=fusion_emb.cuda()
    time=time.cuda()
    model=MIDTransformer(dim=3)
    model.cuda()
    
    from thop import profile
    with torch.no_grad():
        flops, params = profile(model, inputs=(x,x))
        print(f"Total FLOPS: {flops / 1e9:.2f} G")  # 转换为 Giga-FLOPS
        print(f"Total params from thop: {params / 1e6:.2f} M")  # 转换为 Million
        print("finish")




