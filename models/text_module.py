import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat
from transformers import RobertaModel, RobertaTokenizerFast
import math


class PositionEmbeddingSine2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: Tensor, hidden_dim: int):
        """
        @param mask: a tensor of shape [B, H, W]
        @param hidden_dim: int
        @return:
        """
        num_pos_feats = hidden_dim // 2

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features.clone())
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output
    

class Text_projector(nn.Module):
    def __init__(self, num_encoder_layers=3, num_decoder_layers=3,
                 text_encoder_type="roberta-base", freeze_text_encoder=True, d_model=512):
        super().__init__()
        self.d_model=d_model
        self.text_encoder = RobertaModel.from_pretrained('/media/cygzz/data/guanjl/Code_self/roberta-base/')
        self.text_encoder.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
        # self.tokenizer = RobertaTokenizerFast.from_pretrained("/home/whq/HKUSTGZ/RVOS/MTTR-Miccai/roberta-base")   #
        self.tokenizer = RobertaTokenizerFast.from_pretrained('/media/cygzz/data/guanjl/Code_self/roberta-base/')
        # print("self.tokenizer=",type(self.tokenizer))
        # print("self.tokenizer=",self.tokenizer)
        self.freeze_text_encoder = freeze_text_encoder   
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.txt_proj = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=self.d_model,
            dropout=0.5,
        )
        self.txt_proj_w = FeatureResizer(
            input_feat_size=512,
            output_feat_size=self.d_model,
            dropout=0.5,
        )
    
    def forward(self, text_queries):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        # print("tokenized_queries.type=",type(tokenized_queries))
        #tokenized_queries = tokenized_queries.cuda()
        tokenized_queries = {key: value.cuda() for key, value in tokenized_queries.items()}
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries)
        # Transpose memory because pytorch's attention expects sequence first
        txt_memory = rearrange(encoded_text.last_hidden_state, 'b s c -> s b c')
        # print("txt_memory_ori.shape=",txt_memory.shape)
        txt_memory = self.txt_proj(txt_memory)  # change text embeddings dim to model dim
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        txt_pad_mask = tokenized_queries['attention_mask'].ne(1).bool() 
        txt_memory=txt_memory.transpose(1,0) # [B, S]
        return txt_memory



class Text_projector_v1(nn.Module):
    def __init__(self, num_encoder_layers=3, num_decoder_layers=3,
                 text_encoder_type="roberta-base", freeze_text_encoder=True):
        super().__init__()
        
        self.text_encoder = RobertaModel.from_pretrained('/media/cygzz/data/guanjl/Code_self/roberta-base/')
        self.text_encoder.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
        # self.tokenizer = RobertaTokenizerFast.from_pretrained("/home/whq/HKUSTGZ/RVOS/MTTR-Miccai/roberta-base")   #
        self.tokenizer = RobertaTokenizerFast.from_pretrained('/media/cygzz/data/guanjl/Code_self/roberta-base/')
        # print("self.tokenizer=",type(self.tokenizer))
        # print("self.tokenizer=",self.tokenizer)
        self.freeze_text_encoder = freeze_text_encoder   
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        
    
    def forward(self, text_queries):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        # print("tokenized_queries.type=",type(tokenized_queries))
        #tokenized_queries = tokenized_queries.cuda()
        tokenized_queries = {key: value.cuda() for key, value in tokenized_queries.items()}
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries)
        # Transpose memory because pytorch's attention expects sequence first
        txt_memory = rearrange(encoded_text.last_hidden_state, 'b s c -> s b c')
        # print("txt_memory_ori.shape=",txt_memory.shape)
        
        return txt_memory


class Text_projector_v2(nn.Module):
    def __init__(self, num_encoder_layers=3, num_decoder_layers=3,
                 text_encoder_type="roberta-base", freeze_text_encoder=True, d_model=512):
        super().__init__()
        self.d_model=d_model
        
        self.freeze_text_encoder = freeze_text_encoder   
        

        self.txt_proj = FeatureResizer(
            input_feat_size=768,
            output_feat_size=self.d_model,
            dropout=0.5,
        )
        self.txt_proj_w = FeatureResizer(
            input_feat_size=512,
            output_feat_size=self.d_model,
            dropout=0.5,
        )
    
    def forward(self, txt_memory_ori):
       
        # print("txt_memory_ori.shape=",txt_memory.shape)
        txt_memory = self.txt_proj(txt_memory_ori)  # change text embeddings dim to model dim
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
       
        return txt_memory


class Text_projector_v3(nn.Module):
    def __init__(self, num_encoder_layers=3, num_decoder_layers=3,
                 text_encoder_type="roberta-base", freeze_text_encoder=True, d_model=512):
        super().__init__()
        self.d_model=d_model
        self.text_encoder = RobertaModel.from_pretrained('/media/cygzz/data/guanjl/Code_self/roberta-base/')
        self.text_encoder.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
        # self.tokenizer = RobertaTokenizerFast.from_pretrained("/home/whq/HKUSTGZ/RVOS/MTTR-Miccai/roberta-base")   #
        self.tokenizer = RobertaTokenizerFast.from_pretrained('/media/cygzz/data/guanjl/Code_self/roberta-base/')
        # print("self.tokenizer=",type(self.tokenizer))
        # print("self.tokenizer=",self.tokenizer)
        self.freeze_text_encoder = freeze_text_encoder   
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        
    
    def forward(self, text_queries):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        # print("tokenized_queries.type=",type(tokenized_queries))
        #tokenized_queries = tokenized_queries.cuda()
        tokenized_queries = {key: value.cuda() for key, value in tokenized_queries.items()}
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries)
        # Transpose memory because pytorch's attention expects sequence first
        txt_memory = rearrange(encoded_text.last_hidden_state, 'b s c -> s b c')
        # print("txt_memory_ori.shape=",txt_memory.shape)
        #xt_memory = self.txt_proj(txt_memory)  # change text embeddings dim to model dim
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        txt_pad_mask = tokenized_queries['attention_mask'].ne(1).bool() 
        txt_memory=txt_memory.transpose(1,0) # [B, S]
        return txt_memory,txt_pad_mask  
    
if __name__=="__main__":
    from thop import profile
    text=["i love you"]
    #text=text.cuda()
    model=Text_projector_v3()
    model=model.cuda()
    model.eval()
    a=model(text)
    print("finish")
    flops, params = profile(model, inputs=(text,))
    print(f"Total FLOPS: {flops / 1e9:.2f} G")  # 转换为 Giga-FLOPS
    print(f"Total params from thop: {params / 1e6:.2f} M")  # 转换为 Million

