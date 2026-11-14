import torch
import torch.nn as nn
import sys
sys.path.append('/media/cygzz/data/guanjl/Code_self/models')
from basic_model import *
from Image_Text_Fusion import *
from torch.nn.init import _calculate_fan_in_and_fan_out
class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

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

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output

class ResUnet_V2(nn.Module):
    def __init__(self, channel,filters=[64, 128, 256, 512]):
        super(ResUnet_V2, self).__init__()

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

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], channel, 1, 1),
            nn.Sigmoid(),
        )
        self.Fusion=MIDTransformer(filters[-1])
        self.fusion_before=ResidualConv(filters[-1],filters[-1],1,1)
        self.fusion_after=ResidualConv(filters[-1],filters[-1],1,1)

    def forward(self, x, time_emb,fusion_emb):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1,time_emb)
        x3 = self.residual_conv_2(x2,time_emb)
        x4 = self.residual_conv_3(x3,time_emb)
        
        #Fusion
        x_in=self.fusion_before(x4,time_emb)
        middle_out=self.Fusion(x_in,fusion_emb)
        x_out=self.fusion_after(middle_out,time_emb)
       
        # Decode
        x5 = self.upsample_1(x_out)
        x6 = self.up_residual_conv1(x5,time_emb)

        x7 = self.upsample_2(x6)
        x8 = self.up_residual_conv2(x7,time_emb)

        x9 = self.upsample_3(x8)
        x10 = self.up_residual_conv3(x9,time_emb)

        output = self.output_layer(x10)

        return output

class ResUnet_V3(nn.Module):
    def __init__(self, channel,filters=[64, 128, 256, 512]):
        super(ResUnet_V3, self).__init__()

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

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3]+filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2]+filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1]+filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], channel, 1, 1),
            nn.Sigmoid()
          
        )
        # self.Fusion=MIDTransformer(filters[-1])
        # self.fusion_before=ResidualConv(filters[-1],filters[-1],1,1)
        # self.fusion_after=ResidualConv(filters[-1],filters[-1],1,1)

    def forward(self, x, time_emb=None,fusion_emb=None):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)#224
        x2 = self.residual_conv_1(x1,time_emb)#112
        x3 = self.residual_conv_2(x2,time_emb)#56
        x4 = self.residual_conv_3(x3,time_emb)#28
        
        #Fusion
        # x_in=self.fusion_before(x4,time_emb)
        # middle_out=self.Fusion(x_in,fusion_emb)
        # x_out=self.fusion_after(middle_out,time_emb)
       
        # Decode
        x5 = self.upsample_1(x4)#56
        x5 = torch.cat([x5,x3],dim=1)
        x6 = self.up_residual_conv1(x5,time_emb)

        x7 = self.upsample_2(x6)#112
        x7 = torch.cat([x7,x2],dim=1)
        x8 = self.up_residual_conv2(x7,time_emb)

        x9 = self.upsample_3(x8)#224
        x9 = torch.cat([x9,x1],dim=1)
        x10 = self.up_residual_conv3(x9,time_emb)

        output = self.output_layer(x10)

        return output

class ResUnet_V3_label(nn.Module):
    def __init__(self, channel,num_classes=10,filters=[64, 128, 256, 512]):
        super(ResUnet_V3_label, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )
        self.label_embedding = nn.Embedding(num_classes, 128)

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3]+filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2]+filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1]+filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], channel, 1, 1),
           
        )
        #self.Fusion=MIDTransformer(filters[-1])
        self.fusion_before=ResidualConv(filters[-1],filters[-1],1,1)
        self.fusion_after=ResidualConv(filters[-1],filters[-1],1,1)

    def forward(self, x, y,time_emb=None,fusion_emb=None):
        # Encode
        if y is not None:
            label_emb = self.label_embedding(y)
            label_emb = label_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, label_emb], dim=1)
        x1 = self.input_layer(x) + self.input_skip(x)#224
        x2 = self.residual_conv_1(x1,time_emb)#112
        x3 = self.residual_conv_2(x2,time_emb)#56
        x4 = self.residual_conv_3(x3,time_emb)#28
        
        #Fusion
        # x_in=self.fusion_before(x4,time_emb)
        # middle_out=self.Fusion(x_in,fusion_emb)
        # x_out=self.fusion_after(middle_out,time_emb)
       
        # Decode
        x5 = self.upsample_1(x_4)#56
        x5 = torch.cat([x5,x3],dim=1)
        x6 = self.up_residual_conv1(x5,time_emb)

        x7 = self.upsample_2(x6)#112
        x7 = torch.cat([x7,x2],dim=1)
        x8 = self.up_residual_conv2(x7,time_emb)

        x9 = self.upsample_3(x8)#224
        x9 = torch.cat([x9,x1],dim=1)
        x10 = self.up_residual_conv3(x9,time_emb)

        output = self.output_layer(x10)

        return output
    
class ResUnet_V4(nn.Module):
    def __init__(self, channel,filters=[64, 128, 256, 512]):
        super(ResUnet_V4, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv_Fusion(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv_Fusion(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv_Fusion(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv_Fusion(filters[3]+filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv_Fusion(filters[2]+filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv_Fusion(filters[1]+filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], channel, 1, 1),
            nn.Sigmoid(),
        )
        self.Fusion=MIDTransformer(filters[-1])
        self.fusion_before=ResidualConv(filters[-1],filters[-1],1,1)
        self.fusion_after=ResidualConv(filters[-1],filters[-1],1,1)

    def forward(self, x, text_emb,time_emb=None,fusion_emb=None):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)#224
        x2 = self.residual_conv_1(x1,text_emb,time_emb)#112
        x3 = self.residual_conv_2(x2,text_emb,time_emb)#56
        x4 = self.residual_conv_3(x3,text_emb,time_emb)#28
        
        #Fusion
        x_in=self.fusion_before(x4,time_emb)
        middle_out=self.Fusion(x_in,fusion_emb)
        x_out=self.fusion_after(middle_out,time_emb)
       
        # Decode
        x5 = self.upsample_1(x_out)#56
        x5 = torch.cat([x5,x3],dim=1)
        x6 = self.up_residual_conv1(x5,text_emb,time_emb)

        x7 = self.upsample_2(x6)#112
        x7 = torch.cat([x7,x2],dim=1)
        x8 = self.up_residual_conv2(x7,text_emb,time_emb)

        x9 = self.upsample_3(x8)#224
        x9 = torch.cat([x9,x1],dim=1)
        x10 = self.up_residual_conv3(x9,text_emb,time_emb)

        output = self.output_layer(x10)

        return output

class ResUnet_V5(nn.Module):
    def __init__(self, channel,filters=[64, 128, 256, 512]):
        super(ResUnet_V5, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel+3, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel+3, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.input_layertt = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skiptt = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1tt = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2tt = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3tt = ResidualConv(filters[2], filters[3], 2, 1)


        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3]+filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2]+filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1]+filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid()
          
        )
        self.Fusion=MIDTransformer(filters[-1])
        self.fusion_before=ResidualConv(filters[-1],filters[-1],1,1)
        self.fusion_after=ResidualConv(filters[-1],filters[-1],1,1)

    def forward(self, x, img,time_emb=None,fusion_emb=None):
        # Encode
        x1t = self.input_layertt(img) + self.input_skiptt(img)#224
        x2t = self.residual_conv_1tt(x1t,time_emb)#112
        x3t = self.residual_conv_2tt(x2t,time_emb)#56
        x4t = self.residual_conv_3tt(x3t,time_emb)#28

        x=torch.cat([x,img],dim=1)
        x1 = self.input_layer(x) + self.input_skip(x)#224
        x1 = x1 +x1t
        x2 = self.residual_conv_1(x1,time_emb)#112
        x2 = x2+x2t
        x3 = self.residual_conv_2(x2,time_emb)#56
        x3 = x3+x3t
        x4 = self.residual_conv_3(x3,time_emb)#28
        x4 = x4+x4t
        
        #Fusion
        x_in=self.fusion_before(x4,time_emb)
        middle_out=self.Fusion(x_in,fusion_emb)
        x_out=self.fusion_after(middle_out,time_emb)
       
        # Decode
        x5 = self.upsample_1(x_out)#56
        x5 = torch.cat([x5,x3],dim=1)
        x6 = self.up_residual_conv1(x5,time_emb)

        x7 = self.upsample_2(x6)#112
        x7 = torch.cat([x7,x2],dim=1)
        x8 = self.up_residual_conv2(x7,time_emb)

        x9 = self.upsample_3(x8)#224
        x9 = torch.cat([x9,x1],dim=1)
        x10 = self.up_residual_conv3(x9,time_emb)

        output = self.output_layer(x10)

        return output


            
class Condition_Encoder(nn.Module):
    def __init__(self, channel=3,d_model=512,filters=[64, 128, 256, 512]):
        super(Condition_Encoder, self).__init__()
        
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
        x1 = self.input_layer(img) + self.input_skip(img)
        
        x2 = self.residual_conv_1(x1,time_emb)
        x2 = self.fusion2(x2,text_emb)
        x3 = self.residual_conv_2(x2,time_emb)
        x3 = self.fusion3(x3,text_emb)
        x4 = self.residual_conv_3(x3,time_emb)
        fusion_out=self.fusion4(x4,text_emb)

        return fusion_out

class con_coder(nn.Module):
    def __init__(self,channel=3,filters=[64, 128, 256, 512]):
        super(con_coder,self).__init__()
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
    def forward(self,img):
        cc_list=[]
        x1 = self.input_layer(img) + self.input_skip(img)
        cc_list.append(x1)
        
        x2 = self.residual_conv_1(x1)
        cc_list.append(x2)
        
        x3 = self.residual_conv_2(x2)
        cc_list.append(x3)
       
        x4 = self.residual_conv_3(x3)
        cc_list.append(x4)
        

        return cc_list

class classifier(nn.Module):
   def __init__(self, filters=[64, 128, 256, 512]):
        super(classifier, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(4, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(4, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.lin1=nn.Linear(filters[3],filters[1])
        self.lin2=dense(filters[1],2)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
        self.layer=nn.AdaptiveAvgPool2d((1, 1))
    
   def forward(self, x_t,x,time_emb=None):
        #time_emb=self.time_mlp(time_emb)
        # Encode
        img=torch.cat((x_t,x),dim=1)
        x1 = self.input_layer(img) + self.input_skip(img)
        x2 = self.residual_conv_1(x1,time_emb)
        x3 = self.residual_conv_2(x2,time_emb)
        x4 = self.residual_conv_3(x3,time_emb)
        x4=self.layer(x4)
        x4=x4.view(x_t.size()[0],512)
        out=self.lin1(x4)
        out=self.lin2(out)
        return out
   
class Discriminator(nn.Module):
   def __init__(self, filters=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(2, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.lin1=nn.Linear(filters[3],filters[1])
        self.lin2=dense(filters[1],1)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128),
                                      nn.Linear(128, 512),
                                      nn.GELU(),
                                      nn.Linear(512, 512))
        self.layer=nn.AdaptiveAvgPool2d((1, 1))
    
   def forward(self, x_t,x,time_emb=None):
        time_emb=self.time_mlp(time_emb)
        # Encode
        img=torch.cat([x_t,x],dim=1)
        x1 = self.input_layer(img) + self.input_skip(img)
        x2 = self.residual_conv_1(x1,time_emb)
        x3 = self.residual_conv_2(x2,time_emb)
        x4 = self.residual_conv_3(x3,time_emb)
        x4 = self.layer(x4)
        x4=x4.view(-1,512)
        out=self.lin1(x4)
        out=self.lin2(out)
        return out

def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin

if __name__=="__main__":
    x=torch.zeros((1,3,224,224))
    fusion_emb=torch.zeros((1,512,28,28))
    time=torch.zeros((1,512))
    
    x=x.cuda()
    fusion_emb=fusion_emb.cuda()
    time=time.cuda()
    model=ResUnet_V3(channel=3,filters=[64, 128, 256, 512])
    model.cuda()
    out=model(x,time,fusion_emb)
    from thop import profile
    flops, params = profile(model, inputs=(x,time,fusion_emb))
    print(f"Total FLOPS: {flops / 1e9:.2f} G")  # 转换为 Giga-FLOPS
    print(f"Total params from thop: {params / 1e6:.2f} M")  # 转换为 Million
    print("finish")
