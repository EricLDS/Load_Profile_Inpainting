import time
import torch
import config
from config import model, opt
from dataset import trainloader
from model import DisGIN
from loss import SNGenLoss, SNDisLoss
from utils import Plot_Helper

disc = DisGIN(n_fea=config.NF_DIS)
opt_disc = torch.optim.Adam(disc.parameters(), lr=config.LR, betas=(0.9, 0.99))
l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
gan_loss = SNGenLoss()
dis_loss = SNDisLoss()

if config.CUDA:
    model.cuda()
    disc.cuda()

plot_helper = Plot_Helper(config)
 
# ------------------------------------Training------------------------------------
start_t = time.time()
for epoch in range(config.N_EPOCH):
    for i, data in enumerate(trainloader):
        model_input, mask, gt = data
        loss_train_rec = []
         
        # train discriminator
        opt_disc.zero_grad()
        
        pre, pre_coarse,_ = model(model_input)
            
        
        if config.USE_LOCAL_GAN_LOSS:
            idx_st = int(config.DIM_INPUT/24*9.5)
            idx_end = idx_st + int(config.DIM_INPUT/24*5)
            pos = torch.cat([gt[:,:,idx_st:idx_end],
                             mask[:,:,idx_st:idx_end],
                             torch.full_like(mask[:,:,idx_st:idx_end], 1.)], dim=1)
            neg = torch.cat([pre[:,:,idx_st:idx_end],
                             mask[:,:,idx_st:idx_end],
                             torch.full_like(mask[:,:,idx_st:idx_end], 1.)], dim=1)
            pos_neg = torch.cat([pos, neg], dim=0)
        else:
            pos = torch.cat([gt, mask, torch.full_like(mask, 1.)], dim=1)
            neg = torch.cat([pre, mask, torch.full_like(mask, 1.)], dim=1)
            pos_neg = torch.cat([pos, neg], dim=0)
          
        pred_pos_neg, _ = disc(pos_neg)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        d_loss = dis_loss(pred_pos, pred_neg)
        d_loss.backward(retain_graph=True)
        opt_disc.step()
          
        # train generator
        opt_disc.zero_grad()
        disc.zero_grad()
        opt.zero_grad()
        model.zero_grad()
        
        pred_neg, fea_neg = disc(neg) # Fake samples
        _, fea_pos = disc(pos) # Fake samples
        
        g_loss = gan_loss(pred_neg)
        fea_loss = mse_loss(fea_pos, fea_neg)
        r_loss = l1_loss(gt, pre)
        r_loss_coarse = l1_loss(gt, pre_coarse)
        loss_train = r_loss_coarse + config.W_GAN * g_loss +\
            config.W_P2P * r_loss + config.W_FEA * fea_loss
            
        loss_train.backward()
        opt.step()
     
        plot_helper.record_training(loss_train.cpu().detach().numpy())
        

    if epoch % config.SAVE_PER_EPO == 0:
        model.save_checkpoint(epoch)
        msg_tmp = "---epoch:{}".format(epoch)
        msg_tmp += " || Time(s):{}".format(int(time.time() - start_t))
        print(msg_tmp)

