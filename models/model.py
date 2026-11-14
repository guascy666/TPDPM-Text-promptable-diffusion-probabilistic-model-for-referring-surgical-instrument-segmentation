import sys
sys.path.append('/media/cygzz/data/guanjl/Code_self/models')
from basic_model import *
from resunet import *
from Image_Text_Fusion import *
import torch
import torch.nn as nn
import numpy as np
from text_module import *
from beartype import beartype
from collections import namedtuple
from functools import partial
from random import random
from tqdm.auto import tqdm

ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])

# 判断变量是否存在
def exists(x):
    return x is not None


# 变量的默认选择
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

# 创建学习率更新策略
def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def func(x):
        """
        学习率调整函数
        根据 step 数返回一个学习率倍率因子，注意在训练开始之前，pytorch 会提前调用一次 lr_scheduler.step() 方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup 过程中 lr 倍率因子从 warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup 后 lr 倍率因子从 1 -> 0
            # 参考 deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)

# 高斯扩散训练器类
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# 线性采样方案
def linear_beta_schedule(time_steps):
    scale = 1000 / time_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, time_steps, dtype=torch.float64)


# 正弦采样方案
def cosine_beta_schedule(time_steps, s=0.008):
    steps = time_steps + 1
    x = torch.linspace(0, time_steps, steps, dtype=torch.float64)
    alphas_cum_prod = torch.cos(((x / time_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cum_prod = alphas_cum_prod / alphas_cum_prod[0]
    betas = 1 - (alphas_cum_prod[1:] / alphas_cum_prod[:-1])
    return torch.clip(betas, 0, 0.999)

class GUA(nn.Module):
    def __init__(self,time_steps=1000, choice='v1',sampling_time_steps=None, objective='predict_x0',
                 beta_schedule='cosine', ddim_sampling_eta=1.):
        """
        :param time_steps: 加噪的步长
        :param sampling_time_steps: 采样步长
        :param objective: 预测目标
        :param beta_schedule: 加噪方案
        :param ddim_sampling_eta: 采样率
        """
        super().__init__()
        if choice=='v1':
            self.backbone=ResUnet(channel=3)
        elif choice=='v2':
            self.backbone=ResUnet_V2(channel=3)
        elif choice=='v3':
            self.backbone=ResUnet_V3(channel=3)
        elif choice=='v4':
            self.backbone=ResUnet_V4(channel=3)
        #self.backbone=ResUnet_V2(channel=3)
        self.choice=choice
        self.fusion=Condition_Encoder(channel=3)
        self.text_encoder=Text_projector_v2()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
        self.objective = objective
        

        # 加噪方案的选择, 获取 beta
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(time_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(time_steps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 获取 alpha 值和累乘结果
        alphas = 1. - betas
        alphas_cum_prod = torch.cumprod(alphas, dim=0)
        alphas_cum_prod_prev = F.pad(alphas_cum_prod[:-1], (1, 0), value=1.)

        # 根据加噪长度 获取加噪步长和采样步长
        time_steps, = betas.shape
        self.num_time_steps = int(time_steps)
        self.sampling_time_steps = default(sampling_time_steps, time_steps)
        assert self.sampling_time_steps <= time_steps

        # 默认采样时间步数到训练时的时间步数
        self.is_ddim_sampling = self.sampling_time_steps < time_steps
        self.ddim_sampling_eta = ddim_sampling_eta

        self.num_timesteps = time_steps
        beta_start=0.0001
        beta_end=0.02
        self.betas = torch.linspace(beta_start, beta_end, time_steps)
        self.betas.cuda()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        

        # 辅助函数，用于将缓冲区从 float64 注册到 float32
        
       
        

    def device(self):
        return self.module.device()
        
    
    def forward_text(self,text):
        text_emb= self.text_encoder(text)
        return text_emb
    
    def forward_fusion(self,image,text,time):
        return self.fusion(image,text,time)


    

    def model_predictions(self, x, t, c , text):
        """ 模型的预测过程 """

        # 获取 UNet 网络的输出
        time = self.time_mlp(t)
        text_emb=self.forward_text(text)
        fusion_emb=self.forward_fusion(c,text_emb,time)
        if self.choice=='v4':
            model_output=self.backbone(x,text_emb,time,fusion_emb)
        else:
            model_output = self.backbone(x, time, fusion_emb)

       

        

        return model_output

    

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0)
        noise=noise.cuda()
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alphas_cumprod_t=sqrt_alphas_cumprod_t.cuda()
        sqrt_one_minus_alphas_cumprod_t=sqrt_one_minus_alphas_cumprod_t.cuda()
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def p_sample(self, x_t, t,cond,text,tk):
      
        # noise_pred = self.model_predictions(x_t,t,cond,text)
        # f1=self.alphas[t]
        # f1.cuda()
        # f2=self.alphas_cumprod[t]
        # f2.cuda()
        # f3=self.alphas_cumprod_prev[t]
        # f3.cuda()
        # f4=self.betas[t]
        # f4.cuda()
        # f5 = self.posterior_variance[t]
        # f5=f5.cuda()
        # t=t.cuda()
        t=t.cuda()
        x0_pre = self.model_predictions(x_t,t,cond,text)
        if t.cpu()==0:
            return x0_pre
        if t.cpu()-1>0:
            x_prev,_=self.forward_diffusion(x0_pre,t.cpu()-1)
            xk=x_prev.detach()
            return xk
        else:
            return x0_pre
        # posterior_mean_coef1 = (torch.sqrt(self.alphas[t]) * (1. - self.alphas_cumprod_prev[t])) / (
        #         1. - self.alphas_cumprod[t])
        # posterior_mean_coef2 = (torch.sqrt(self.alphas_cumprod_prev[t]) * self.betas[t]) / (
        #         1. - self.alphas_cumprod[t])
       
      
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def sample(self, shape,cond,text):
        #batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        total_timesteps=200
        sampling_timesteps=50
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape)
        batch=shape[0]
        img=img.cuda()
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, dtype = torch.long)
            time_cond = time_cond.cuda()
            #self_cond = x_start if self.self_condition else None
            x_start= self.model_predictions(img, time_cond, cond,text)
            pred_noise=self.pred_noise()

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)
    

    def p_losses(self, x_start, t, cond,text):
        """ 损失计算过程 """

        # 根据噪声生成加噪后图像
        
        x_t,noise=self.forward_diffusion(x_start,t)
        t=t.cuda()
        #x = self.q_sample(x_start=x_start, t=t, noise=noise_g)
        # 预测并采取梯度步长
        pred_noise = self.model_predictions(x_t, t, cond, text)

        # 选择预测目标
       

        # 计算损失 MSE
        return F.mse_loss(pred_noise, x_start)

    def forward(self, img, cond_img,text):
        """ 前向计算过程, 直接获取损失 
        img是mask,复制通道1->3
        cond_img是原图
        text是文本输入
        """

        # 数据格式转换
        if img.ndim == 3:
            img = rearrange(img, 'b h w -> b 1 h w')

        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, 'b h w -> b 1 h w')

        # 获取运行设备
        #device = self.device()

        # 将输入和条件图像添加到运行设备
        img, cond_img ,text = img.cuda(), cond_img.cuda(),text.cuda()
        #print(device)

        # 对图像的大小进行判断, 并给出警告
        b, c, h, w = img.shape
        
    

       

        # 生成时间编码
        times = torch.randint(0, self.num_time_steps, (b,)).long()
        #times =times.cuda()

        # 对图像进行归一化
        #img = normalize_to_neg_one_to_one(img)

        # 计算损失
        return self.p_losses(img, times, cond_img,text)

from monai.losses.dice import DiceLoss

def dice_loss(pred,target):
    smooth = 1e-5
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return 1-(2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    #return loss
class GUA_new(nn.Module):
    def __init__(self,time_steps=1000, choice='v1',sampling_time_steps=None, objective='predict_x0',
                 beta_schedule='cosine', ddim_sampling_eta=1.):
        """
        :param time_steps: 加噪的步长
        :param sampling_time_steps: 采样步长
        :param objective: 预测目标
        :param beta_schedule: 加噪方案
        :param ddim_sampling_eta: 采样率
        """
        super().__init__()
      
        self.backbone=ResUnet_V5(channel=1)
        #self.backbone=ResUnet_V2(channel=3)
        self.choice=choice
        self.fusion=Condition_Encoder(channel=3)
        self.text_encoder=Text_projector_v2()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
        self.objective = objective
        

        # 加噪方案的选择, 获取 beta
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(time_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(time_steps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 获取 alpha 值和累乘结果
        alphas = 1. - betas
        alphas_cum_prod = torch.cumprod(alphas, dim=0)
        alphas_cum_prod_prev = F.pad(alphas_cum_prod[:-1], (1, 0), value=1.)

        # 根据加噪长度 获取加噪步长和采样步长
        time_steps, = betas.shape
        self.num_time_steps = int(time_steps)
        self.sampling_time_steps = default(sampling_time_steps, time_steps)
        assert self.sampling_time_steps <= time_steps

        # 默认采样时间步数到训练时的时间步数
        self.is_ddim_sampling = self.sampling_time_steps < time_steps
        self.ddim_sampling_eta = ddim_sampling_eta

        self.num_timesteps = time_steps
        beta_start=0.0001
        beta_end=0.02
        self.betas = torch.linspace(beta_start, beta_end, time_steps)
        self.betas.cuda()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        #register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. -self. alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - self.alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        #register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod))
        

        # 辅助函数，用于将缓冲区从 float64 注册到 float32
        
       
        

    def device(self):
        return self.module.device()
        
    
    def forward_text(self,text):
        text_emb= self.text_encoder(text)
        return text_emb
    
    def forward_fusion(self,image,text,time):
        return self.fusion(image,text,time)


    

    def model_predictions(self, x, t, c , text):
        """ 模型的预测过程 """

        # 获取 UNet 网络的输出
        time = self.time_mlp(t)
        text_emb=self.forward_text(text)
        fusion_emb=self.forward_fusion(c,text_emb,time)
        
        model_output = self.backbone(x, c,time, fusion_emb)

       

        

        return model_output

    

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0)
        noise=noise.cuda()
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alphas_cumprod_t=sqrt_alphas_cumprod_t.cuda()
        sqrt_one_minus_alphas_cumprod_t=sqrt_one_minus_alphas_cumprod_t.cuda()
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def p_sample(self, x_t, t,cond,text,tk):
        t=t.cuda()
        x0_pre = self.model_predictions(x_t,t,cond,text)
        x_prev,_=self.forward_diffusion(x0_pre,t.cpu()-1)
        return x0_pre,x_prev

    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def sample(self, shape,cond,text):
        #batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        total_timesteps=200
        sampling_timesteps=50
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape)
        batch=shape[0]
        img=img.cuda()
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, dtype = torch.long)
            time_cond = time_cond.cuda()
            #self_cond = x_start if self.self_condition else None
            x_start= self.model_predictions(img, time_cond, cond,text)
            pred_noise=self.pred_noise()

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            eta=0
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img 

        #ret = self.unnormalize(ret)
        #ret = torch.sigmoid(ret)
        return ret
    

    def p_losses(self, x_start, t, cond,text):
        """ 损失计算过程 """

        # 根据噪声生成加噪后图像
        
        x_t,noise=self.forward_diffusion(x_start,t)
        t=t.cuda()
        #x = self.q_sample(x_start=x_start, t=t, noise=noise_g)
        # 预测并采取梯度步长
        pred_noise = self.model_predictions(x_t, t, cond, text)

        # 选择预测目标
       

        # 计算损失 MSE
        return F.mse_loss(pred_noise, x_start).mean()+dice_loss(pred_noise,x_start)

    def forward(self, img, cond_img,text):
        """ 前向计算过程, 直接获取损失 
        img是mask,复制通道1->3
        cond_img是原图
        text是文本输入
        """

        # 数据格式转换
        if img.ndim == 3:
            img = rearrange(img, 'b h w -> b 1 h w')

        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, 'b h w -> b 1 h w')

        # 获取运行设备
        #device = self.device()

        # 将输入和条件图像添加到运行设备
        img, cond_img ,text = img.cuda(), cond_img.cuda(),text.cuda()
        #print(device)

        # 对图像的大小进行判断, 并给出警告
        b, c, h, w = img.shape
        
    

       

        # 生成时间编码
        times = torch.randint(0, self.num_time_steps, (b,)).long()
        #times =times.cuda()

        # 对图像进行归一化
        #img = normalize_to_neg_one_to_one(img)

        # 计算损失
        return self.p_losses(img, times, cond_img,text)
    
class GUA_V2(nn.Module):
    def __init__(self,time_steps=1000, choice='v1',sampling_time_steps=None, objective='predict_x0',
                 beta_schedule='cosine', ddim_sampling_eta=1.):
        """
        :param time_steps: 加噪的步长
        :param sampling_time_steps: 采样步长
        :param objective: 预测目标
        :param beta_schedule: 加噪方案
        :param ddim_sampling_eta: 采样率
        """
        super().__init__()
        if choice=='v1':
            self.backbone=ResUnet(channel=3)
        elif choice=='v2':
            self.backbone=ResUnet_V2(channel=3)
        elif choice=='v3':
            self.backbone=ResUnet_V3_label(channel=3)
        elif choice=='v4':
            self.backbone=ResUnet_V4(channel=3)
        #self.backbone=ResUnet_V2(channel=3)
        self.choice=choice
        self.fusion=Condition_Encoder(channel=3)
        self.text_encoder=Text_projector_v2()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
        self.objective = objective
        

        # 加噪方案的选择, 获取 beta
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(time_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(time_steps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 获取 alpha 值和累乘结果
        alphas = 1. - betas
        alphas_cum_prod = torch.cumprod(alphas, dim=0)
        alphas_cum_prod_prev = F.pad(alphas_cum_prod[:-1], (1, 0), value=1.)

        # 根据加噪长度 获取加噪步长和采样步长
        time_steps, = betas.shape
        self.num_time_steps = int(time_steps)
        self.sampling_time_steps = default(sampling_time_steps, time_steps)
        assert self.sampling_time_steps <= time_steps

        # 默认采样时间步数到训练时的时间步数
        self.is_ddim_sampling = self.sampling_time_steps < time_steps
        self.ddim_sampling_eta = ddim_sampling_eta

        self.num_timesteps = time_steps
        beta_start=0.0001
        beta_end=0.02
        self.betas = torch.linspace(beta_start, beta_end, time_steps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        

        # 辅助函数，用于将缓冲区从 float64 注册到 float32
        
       
        

    def device(self):
        return self.module.device()
        
    
    def forward_text(self,text):
        text_emb= self.text_encoder(text)
        return text_emb
    
    def forward_fusion(self,image,text,time):
        return self.fusion(image,text,time)


    

    def model_predictions(self, x, y,t, c , text):
        """ 模型的预测过程 """

        # 获取 UNet 网络的输出
        time = self.time_mlp(t)
        text_emb=self.forward_text(text)
        fusion_emb=self.forward_fusion(c,text_emb,time)
        if np.random.rand() < 0.1:
                noise_pred = self.backbone(x,None,time,fusion_emb)
        else:
                noise_pred = self.backbone(x, y,time,fusion_emb)
           # model_output = self.backbone(x, y,time, fusion_emb)

       

        

        return noise_pred

    

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0)
        noise=noise.cuda()
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alphas_cumprod_t=sqrt_alphas_cumprod_t.cuda()
        sqrt_one_minus_alphas_cumprod_t=sqrt_one_minus_alphas_cumprod_t.cuda()
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def p_sample(self, x_t, t,cond,text,y):
        guidance_scale=3.0
        if t == 0:
            return x_t
        # 无条件预测
        noise_pred_uncond = self.model_predictions(x_t, t=t,c=cond,text=text)
        if y is not None:
            # 有条件预测
            noise_pred_cond = self.model_predictions(x_t, y,t,cond,text)
            # 分类器自由引导
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_uncond
        posterior_mean_coef1 = (torch.sqrt(self.alphas[t]) * (1. - self.alphas_cumprod_prev[t])) / (
                1. - self.alphas_cumprod[t])
        posterior_mean_coef2 = (torch.sqrt(self.alphas_cumprod_prev[t]) * self.betas[t]) / (
                1. - self.alphas_cumprod[t])
        posterior_mean = posterior_mean_coef1 * x_t + posterior_mean_coef2 * noise_pred
        posterior_variance_t = self.posterior_variance[t]
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_prev = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        return x_prev

    def sample(self, shape,cond,text):
        x_T = torch.randn(shape)
        x = x_T
        for t in reversed(range(0, self.num_timesteps)):
            x = self.p_sample(x, t,cond,text)
        return x

    def p_losses(self, x_start, y,t, cond,text):
        """ 损失计算过程 """

        # 根据噪声生成加噪后图像
        
        x_t,noise=self.forward_diffusion(x_start,t)
        t=t.cuda()
        #x = self.q_sample(x_start=x_start, t=t, noise=noise_g)
        # 预测并采取梯度步长
        noise_pred=self.model_predictions(x_t,y,t,cond,text)
        #red_noise = self.model_predictions(x_t,t, cond, text)

        # 选择预测目标
       

        # 计算损失 MSE
        return F.mse_loss(noise_pred, noise)

    def forward(self, img, label,cond_img,text):
        """ 前向计算过程, 直接获取损失 
        img是mask,复制通道1->3
        cond_img是原图
        text是文本输入
        """

        # 数据格式转换
        if img.ndim == 3:
            img = rearrange(img, 'b h w -> b 1 h w')

        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, 'b h w -> b 1 h w')

        # 获取运行设备
        #device = self.device()

        # 将输入和条件图像添加到运行设备
        img, cond_img ,text = img.cuda(), cond_img.cuda(),text.cuda()
        label=label.cuda()
        #print(device)

        # 对图像的大小进行判断, 并给出警告
        b, c, h, w = img.shape
        
    

       

        # 生成时间编码
        times = torch.randint(0, self.num_time_steps, (b,)).long()
        #times =times.cuda()

        # 对图像进行归一化
        #img = normalize_to_neg_one_to_one(img)

        # 计算损失
        return self.p_losses(img, label,times, cond_img,text)

if __name__=="__main__":
    
    text=["i love you","i hate you"]
    mask=torch.zeros((2,1,448,448))
    mask=mask.repeat(1, 3, 1, 1)
    image=torch.zeros((2,3,448,448))
    #text=text.cuda()
    model=GUA()
    model=model.cuda()
    model.train()
    a=model(mask,image,text)
    print("finish")

