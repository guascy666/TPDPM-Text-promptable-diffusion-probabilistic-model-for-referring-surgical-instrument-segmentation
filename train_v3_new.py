import os
import torch
import wandb
import random
import datetime
import torchvision.transforms as transforms
from models.denoise import GaussianDiffusion,Diff4
from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from datasets.datasets import ReferRSVIS,ISICDataset
from torch.utils.data import DataLoader
from models.model import *
import time
#export CUDA_VISIBLE_DEVICES=0,1

def compute(target,pred,smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    
    # 计算交集
    intersection = (pred * target).sum()
    
    # 计算并集
    total = (pred + target).sum()
    union = total - intersection
    
    # 计算IoU
    iou = (intersection + smooth) / (union + smooth)
    
    
    
    # 计算交集
    intersection = (pred * target).sum()
    
    # 计算Dice系数
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return iou,dice

if __name__ == '__main__':

    # Hyper Parameter Setting
    choice='v5'
    dim = 64  # Foundation dimensions of the UNet network
    epochs = 6000  # Iterations
    img_size = 224 # Input Image Size, or 128
    batch_size = 1  # Batch Size, or 16
    save_every = 5 # Iterate X times, save the model
    time_steps = 200  # Time step - number of noise stacks
    adam_beta_1 = 0.950  # Adam Parameter 1
    adam_beta_2 = 0.999  # Adam Parameter 2
    adam_epsilon = 1e-8  # eps value
    weight_decay = 1e-6  # regularization factor
    learning_rate = 2e-5  # learning rate
    mask_channels = 3  # Number of mask channels
    self_condition = False  # self-conditional input
    input_img_channels = 3  
    # Number of input image channels

    # data path
    data_path = "/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18"
    data=data_path.split('/')[-1]
    #load_model_from = '/media/cygzz/data/guanjl/Code_self/output_total/Diff4/2025-04-08 03:03:00_bs8_lr5e-05_size224_step200/checkpoints/state_dict_epoch_75_loss_0.06213918414419097.pt' # Pre-training model paths
    #load_model_from='/media/cygzz/data/guanjl/Code_self/output_total_EndoVis-RS18/Diff4/2025-04-10 01:20:56_bs8_lr5e-05_size224_step200/checkpoints/state_dict_epoch_70_loss_0.0927678746176742.pt'
    # Create path - for saving training results
    load_model_from='/media/cygzz/data/guanjl/Code_self/output_total_Endovis-RS17all/3/Diff4/2025-07-14 01:29:06_bs8_lr5e-05_size224_step200/state_dict_epoch_65_loss_0.08086972534656525.pt'
    
    #os.makedirs(logging_dir, exist_ok=True)

    # Accelerating the training process with gas pedals
    # (gradient accumulation steps, blending accuracy, recording positions)
    accelerator = Accelerator(gradient_accumulation_steps=4,
                              mixed_precision="no",
                              )

    # Establishment of the Wandb project
    if accelerator.is_main_process:
        accelerator.init_trackers("Cold_self")
    #load_model_from=None

    # Network Architecture
    

    

    # Data Loading
 
    dataset_train = ISICDataset('train',dataset_path=data_path)
    dataset_valid = ISICDataset('valid',dataset_path=data_path)

    # Data Processing
    train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True,drop_last=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    # Initializing the Optimizer
    net=Diff4()
    diffusion = GaussianDiffusion(net,(224,224),sampling_timesteps=1)
    name=net.__class__.__name__
    current="output_total"+"_"+data+'/'+name
    os.makedirs(current,exist_ok=True)
    current="output_total"+"_"+data+'/'+name+'/'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_size"+str(img_size)+"_step"+str(time_steps)
    os.makedirs(current,exist_ok=True)

    checkpoint_dir = os.path.join(current,"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create path - for saving training logs
    logging_dir = os.path.join(current,'log.txt')
    diffusion.cuda()
    
    optimizer = AdamW(diffusion.parameters(), lr=learning_rate,
                      betas=(adam_beta_1, adam_beta_2), weight_decay=weight_decay, eps=adam_epsilon)

    # Data acceleration and mixing accuracy settings
    #diffusion,optimizer, train_loader, valid_loader = accelerator.prepare([diffusion, optimizer, train_loader, valid_loader])
    #diffusion = torch.nn.DataParallel(diffusion, device_ids=[0, 1])

    # Training Diffusion Networks
    

    # Loading pre-trained models
    if load_model_from is not None:
        save_dict = torch.load(load_model_from)
        diffusion.load_state_dict(save_dict['model_state_dict'])
        print("loading stat_dict")
        #optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        #accelerator.print(f'Loaded from {load_model_from}')

    # Create a learning rate update strategy, in this case once per step (not per epoch).
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

    # Save information during training and validation
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    best_valid_loss_epoch=-1
    best_valid_iou_epoch=-1
    best_valid_dice_epoch=-1
    best_valid_loss=100000
    best_valid_iou=0
    best_valid_dice=0

    # training process
    for epoch in range(epochs):
        # Recorded losses
        running_train_loss = 0.0
        running_valid_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # Initialize images and masks
        train_img = None
        valid_img = None
        train_mask = None
        valid_mask = None

        # Read image and mask Gradient optimization process
        diffusion.train()
        #rint(diffusion.module.device())
        
        for batch_dict in tqdm(train_loader):
           #with accelerator.accumulate(diffusion):
                
                #Calculating and recording losses
                train_img,train_mask,train_text,_=batch_dict
                
                
                
                #train_mask=train_mask.repeat(1, 3, 1, 1)
                train_mask=train_mask.float()
                train_img=train_img.cuda()
                train_mask=train_mask.cuda()
                train_loss,_ = diffusion(train_mask, train_img, train_text)
                #train_loss=train_loss.mean()

                # backpropagation gradient
                train_loss.backward()

                # Update Gradient
                optimizer.step()
                optimizer.zero_grad()

                # Updated learning rate
                lr_scheduler.step()
                lr = optimizer.param_groups[0]["lr"]

        # Record training loss results
                running_train_loss += train_loss.item() * train_img.size(0)
        epoch_train_loss = running_train_loss / (len(train_loader)*batch_size)
        print('Train Loss : {:.7f}'.format(epoch_train_loss))

        #Model Validation Process
        diffusion.eval()
        if epoch % save_every == 0:
            iou_all=[]
            dice_all=[]
            for batch_dict in tqdm(valid_loader):
                with torch.no_grad():
                   #with accelerator.accumulate(diffusion):
                        # 计算损失并记录
                        valid_img,valid_mask,valid_text,_=batch_dict
                        valid_img=valid_img.cuda()
                        valid_mask=valid_mask.cuda()
                
                        #valid_mask=valid_mask.repeat(1, 3, 1, 1)
                        valid_mask=valid_mask.float()
                        valid_loss ,_= diffusion(valid_mask, valid_img, valid_text)
                        predict_image = diffusion.sample_once(valid_img,valid_text)
                        #valid_loss = valid_loss.mean()

            # Record training loss results
                        running_valid_loss += valid_loss.item() * valid_img.size(0)
                        iou_max=[]
                        dice_max=[]
                    
                        for i in range(len(predict_image[0])):
                            #predict_image1=predict_image[i]
                            iou_item=[]
                            dice_item=[]
                            for j in range(len(predict_image)):
                                predict_images=predict_image[j][i].cpu().detach()

                                #for idx in range(predict_images.shape[0]):
                                    #num+=1
                                save_image = predict_image[j][i].squeeze(0)
                                save_image=torch.sigmoid(save_image)
                                #save_image = (save_image[0]+save_image[1]+save_image[2])/3
                                #save_image=save_image.permute(1,2,0)
                                save_image=save_image.cpu().numpy()
                                #save_image = (save_image[0]+save_image[1]+save_image[2])/3
                                save_image = np.where(save_image >= 0.5, 1, 0)
                                x=save_image
                                
                                
                                #masks=masks.repeat((1,3,1,1))
                                #masks=masks.numpy()
                                save_mask = valid_mask[i].cpu().numpy()
                                save_mask=save_mask[0]
                                #save_mask = save_mask[0]
                                y=save_mask
                                
                                
                                iou,dice=compute(y,x)
                                iou_all.append(iou)
                                dice_all.append(dice)
                                # all_iou.append(iou)
                                # all_dice.append(dice)
            epoch_iou=np.nanmean(iou_all)
            epoch_dice=np.nanmean(dice_all)
            epoch_valid_loss = running_valid_loss / (len(valid_loader)*batch_size)
            print('Valid Loss : {:.7f}'.format(epoch_valid_loss))
            

            if epoch_valid_loss<best_valid_loss:
                    best_valid_loss=epoch_valid_loss
                    best_valid_loss_epoch=epoch
                
            if epoch_iou>best_valid_iou:
                    best_valid_iou=epoch_iou
                    best_valid_iou_epoch=epoch
                
            if epoch_dice>best_valid_dice:
                    best_valid_dice=epoch_dice
                    best_valid_dice_epoch=epoch

        # wandb Record results Training loss Validation loss
        

        # Record the train_loss, valid_loss, and lr for each epoch.
        # with open(results_file, "a") as f:
        #     info = f"[lr: {lr}]\n" \
        #            f"[epoch: {epoch}]\n" \
        #            f"train_loss: {train_loss:.7f}\n" \
        #            f"valid_loss: {valid_loss:.7f}\n"

        #     f.write(info + "\n\n")
        with open(logging_dir, "a") as f:
            info = f"[lr: {lr}]\n" \
                   f"[epoch: {epoch}]\n" \
                   f"train_loss: {epoch_train_loss:.7f}\n" \
                   f"valid_loss: {epoch_valid_loss:.7f}\n" \
                   f"valid_iou: {epoch_iou:.7f}\n" \
                   f"valid_dice: {epoch_dice:.7f}\n" \
                   f"loss epoch: {best_valid_loss_epoch}\n" \
                   f"iou epoch: {best_valid_iou_epoch}\n" \
                   f"dice epoch: {best_valid_dice_epoch}\n" 

            f.write(info + "\n\n")

        # Reasoning process Preservation model
        if epoch % save_every == 0:
            torch.save({'epoch': epoch, 'model_state_dict': diffusion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': valid_loss,
                        }, os.path.join(checkpoint_dir, f'state_dict_epoch_{epoch}_loss_{epoch_valid_loss}.pt'))

            # Getting Sampling Results
            
