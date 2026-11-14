import torch
import torch.nn as nn
import numpy as np

"""
QKV注意力计算
"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
    
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
         # [batch_size, n_heads, len_q, d_v]
        return context


"""
多头注意力
"""
class MultiHeadAttention(nn.Module):
    def __init__(self,channel,d_model,d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()

        #v_im
        self.w1 = nn.Conv2d(in_channels=channel,  # 输入通道数
    out_channels=channel,  # 输出通道数
    kernel_size=1,  # 使用 1x1 卷积核，不改变空间尺寸
    stride=1,  # 步长为 1
    padding=0,  # 不需要填充，因为 1x1 卷积不会改变空间尺寸
    bias=True  # 添加偏置
)
        #v_iq
        self.w2 = nn.Conv2d(in_channels=channel,  # 输入通道数
    out_channels=d_k * n_heads,  # 输出通道数
    kernel_size=1,  # 使用 1x1 卷积核，不改变空间尺寸
    stride=1,  # 步长为 1
    padding=0,  # 不需要填充，因为 1x1 卷积不会改变空间尺寸
    bias=False  # 添加偏置
)
        #Gi
        self.w3 = nn.Conv2d(in_channels=d_k*n_heads,  # 输入通道数
    out_channels=channel,  # 输出通道数
    kernel_size=1,  # 使用 1x1 卷积核，不改变空间尺寸
    stride=1,  # 步长为 1
    padding=0,  # 不需要填充，因为 1x1 卷积不会改变空间尺寸
    bias=False  # 添加偏置
)
        #final
        self.w4 = nn.Conv2d(in_channels=channel,  # 输入通道数
    out_channels=channel,  # 输出通道数
    kernel_size=1,  # 使用 1x1 卷积核，不改变空间尺寸
    stride=1,  # 步长为 1
    padding=0,  # 不需要填充，因为 1x1 卷积不会改变空间尺寸
    bias=False  # 添加偏置
)
        #residual
        self.w5 = nn.Conv2d(in_channels=channel,  # 输入通道数
    out_channels=channel ,  # 输出通道数
    kernel_size=1,  # 使用 1x1 卷积核，不改变空间尺寸
    stride=1,  # 步长为 1
    padding=0,  # 不需要填充，因为 1x1 卷积不会改变空间尺寸
    bias=False  # 添加偏置
)
        
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k, n_heads)
        self.channel=channel
        

    def forward(self, image,text):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        batch_size,channel,hidth,width = image.size()
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        residual=self.w5(image)
        V_im=self.w1(image)
        V_iq=self.w2(image).reshape(batch_size,-1,self.n_heads,hidth,width).transpose(1, 2)
        V_iq=V_iq.reshape(batch_size,self.n_heads,-1,self.d_k)
        L_ik = self.W_K(text).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        #print("Q.SIZE",Q.size())
        L_iv = self.W_V(text).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context= self.ScaledDotProductAttention(V_iq, L_ik, L_iv) #(batchsize,n_heads,len_q,d_v)
        context = context.permute(0,2,1,3)
        context = context.reshape(batch_size,-1,self.n_heads*self.d_v)
        context = context.permute(0,2,1) 
        context = context.reshape(batch_size,self.n_heads *self.d_v,hidth,width)  # context: [batch_size, len_q, n_heads * d_v]
        context = self.w3(context) # [batch_size, len_q, d_model]
        output = self.w4(V_im * context)

        return nn.BatchNorm2d(self.channel).cuda()(output + residual)





"""
Transformer Encoder 子结构
"""
class Encoder(nn.Module):
    def __init__(self,channel=64,d_model=512,d_k=64, d_v=64, n_heads=8):
        super(Encoder,self).__init__()
        self.enc_self_attn = MultiHeadAttention(channel,d_model, d_k, d_v, n_heads)
        
    def forward(self,image,text):
        enc_outputs= self.enc_self_attn(image, text)   # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs

if __name__=="__main__":
    x=torch.zeros((2,3,224,224))
    text=torch.zeros((2,64,756))
    x=x.cuda()
    text=text.cuda()
    model=Encoder(channel=32,d_model=756,d_k=32,d_v=32,n_heads=8)
    model.cuda()
    out=model(x,text)
    print("finish")