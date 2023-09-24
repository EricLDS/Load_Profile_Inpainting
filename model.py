import os
import torch
import torch.nn as nn
from network_module import GatedConv1dWithActivation, GatedDeConv1dWithActivation, SNConvWithActivation
import config


class Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=None):
        super(Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width) # B X C x (*W*H)
        energy = torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width)

        return out, proj_value, attention
        
class MultiHead_Attn(nn.Module):
    """
    Transformer model
    """
    def __init__(self, in_ch=3):
        super(MultiHead_Attn, self).__init__()
        self.in_ch = in_ch
        self.head1 = Attn(in_ch)
        self.head2 = Attn(in_ch)
        self.head3 = Attn(in_ch)
        self.head4 = Attn(in_ch)

        
    def forward(self,x):
        res = self.head1(x)
        res = torch.cat([res, self.head2(x)], dim=1)
        res = torch.cat([res, self.head3(x)], dim=1)
        res = torch.cat([res, self.head4(x)], dim=1)
        return res
    

class GenGIN(torch.nn.Module):
    """
    Generator of Generative Infilling Net with Multi-head Attention
    """
    def __init__(self, in_ch=3, n_fea=64, name='GIN'):
        super(GenGIN, self).__init__()
        self.cor_gc01 = GatedConv1dWithActivation(in_ch, n_fea, 5, 1, 2)
        self.cor_gc02 = GatedConv1dWithActivation(n_fea, 2*n_fea, 4, 2, 1) # //2
        self.cor_gc03 = GatedConv1dWithActivation(2*n_fea, 2*n_fea, 3, 1, 1)
        self.cor_gc04 = GatedConv1dWithActivation(2*n_fea, 4*n_fea, 4, 2, 1) # //2
        self.cor_gc05 = GatedConv1dWithActivation(4*n_fea, 4*n_fea, 3, 1, 1)
        # upsample
        self.cor_gdc1 = GatedDeConv1dWithActivation(2, 4*n_fea, 2*n_fea, 3, 1, 1)
        self.cor_gc09 = GatedConv1dWithActivation(2*n_fea, 2*n_fea, 3, 1, 1)
        self.cor_gdc2 = GatedDeConv1dWithActivation(2, 2*n_fea, n_fea, 3, 1, 1)
        self.cor_gc11 = GatedConv1dWithActivation(n_fea, 1, 3, 1, 1, activation=None)

    
        self.rf1_gc01 = GatedConv1dWithActivation(in_ch, n_fea, 5, 1, 2)
        self.rf1_gc03 = GatedConv1dWithActivation(n_fea, n_fea, 4, 2, 1)
        self.rf1_gc05 = GatedConv1dWithActivation(n_fea, n_fea, 3, 1, 1)
        self.rf1_gc07 = GatedConv1dWithActivation(n_fea, n_fea, 4, 2, 1)

        self.rf2_gc01 = GatedConv1dWithActivation(n_fea, 2*n_fea, 5, 1, 2)
        self.rf2_gc02 = GatedConv1dWithActivation(2*n_fea, 4*n_fea, 3, 1, 1)
        
        self.attn_head1 = Attn(n_fea)
        self.attn_head2 = Attn(n_fea)
        self.attn_head3 = Attn(n_fea)
        self.attn_head4 = Attn(n_fea)

        self.rf_up_gc02 = GatedConv1dWithActivation(8*n_fea, 4*n_fea, 3, 1, 1)
        self.rf_up_gdc1 = GatedDeConv1dWithActivation(2, 4*n_fea, 4*n_fea, 3, 1, 1)
        self.rf_up_gc03 = GatedConv1dWithActivation(4*n_fea, 2*n_fea, 3, 1, 1)
        self.rf_up_gdc2 = GatedDeConv1dWithActivation(2, 2*n_fea, 2*n_fea, 3, 1, 1)
        self.rf_up_gc04 = GatedConv1dWithActivation(2*n_fea, n_fea, 3, 1, 1)
        self.rf_up_gc05 = GatedConv1dWithActivation(n_fea, 1, 3, 1, 1)
        
        self.name = name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def rec_layer(self, nn_layer, x):
        layer_rec = {}
        layer_rec['in'] = x.cpu().detach().numpy()
        x, x_raw, score = nn_layer(x)
        layer_rec['raw'] = x_raw.cpu().detach().numpy()
        layer_rec['out'] = x.cpu().detach().numpy()
        layer_rec['score'] = score.cpu().detach().numpy()
        return x, layer_rec
    
    def forward(self, x):
        masked_x = x[:,0,:].unsqueeze(1)
        mask = x[:,1,:].unsqueeze(1) # 1-hole
        temp = x[:,2,:].unsqueeze(1)
        
        
        x, _, _ = self.cor_gc01(x)
        x, _, _ = self.cor_gc02(x)
        x, _, _ = self.cor_gc03(x)
        x, _, _ = self.cor_gc04(x)
        x, _, _ = self.cor_gc05(x)
        x, _, _ = self.cor_gdc1(x)
        x, _, _ = self.cor_gc09(x)
        x, _, _ = self.cor_gdc2(x)
        x, _, _ = self.cor_gc11(x)

        coarse_x = x
        
        x = torch.cat([masked_x + x*mask, mask, temp], dim=1)
        x, _, _ = self.rf1_gc01(x)
        x, _, _ = self.rf1_gc03(x)
        x, _, _ = self.rf1_gc05(x)
        x, _, _ = self.rf1_gc07(x)
        
        x1, _, _ = self.rf2_gc01(x)
        x1, _, _ = self.rf2_gc02(x1)
        
        x2, _, _ = self.attn_head1(x)
        res = x2
        x2, _, _ = self.attn_head2(x2)
        res = torch.cat([res, x2], dim=1)
        x2, _, _ = self.attn_head3(x2)
        res = torch.cat([res, x2], dim=1)
        x2, _, _ = self.attn_head4(x2)
        res = torch.cat([res, x2], dim=1)
        
        x, _, _ = self.rf_up_gc02(torch.cat([x1, res], dim=1))
        x, _, _ = self.rf_up_gdc1(x)
        x, _, _ = self.rf_up_gc03(x)
        x, _, _ = self.rf_up_gdc2(x)
        x, _, _ = self.rf_up_gc04(x)
        x, _, _ = self.rf_up_gc05(x)
                
        return x, coarse_x, None  
    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_best.h5')
        torch.save(self.state_dict(), filename)


class DisGIN(nn.Module):
    def __init__(self, n_fea=8, name='disc'):
        super(DisGIN, self).__init__()
        if config.USE_LOCAL_GAN_LOSS:
            self.snconv1 = SNConvWithActivation(3, 2*n_fea, 4, 2, 2)
            self.snconv2 = SNConvWithActivation(2*n_fea, 4*n_fea, 4, 2, 2)
            self.snconv3 = SNConvWithActivation(4*n_fea, 8*n_fea, 4, 2, 2)
            self.snconv4 = SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, 2)
            self.snconv5 = SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, 2)
            self.linear = nn.Linear(8*n_fea*2*2, 1)
            
            self.discriminator_net = nn.Sequential(
                SNConvWithActivation(3, 2*n_fea, 4, 2, 2),
                SNConvWithActivation(2*n_fea, 4*n_fea, 4, 2, 2),
                SNConvWithActivation(4*n_fea, 8*n_fea, 4, 2, 2),
                SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, 2),
                SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, 2),
            )
        else:
            self.snconv1 = SNConvWithActivation(3, 2*n_fea, 4, 2, 2)
            self.snconv2 = SNConvWithActivation(2*n_fea, 4*n_fea, 4, 2, 2)
            self.snconv3 = SNConvWithActivation(4*n_fea, 8*n_fea, 4, 2, 2)
            self.snconv4 = SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, 2)
            self.snconv5 = SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, 2)
            self.linear = nn.Linear(8*n_fea*2*2, 1)
            
            self.discriminator_net = nn.Sequential(
                SNConvWithActivation(3, 2*n_fea, 4, 2, 2),
                SNConvWithActivation(2*n_fea, 4*n_fea, 4, 2, 2),
                SNConvWithActivation(4*n_fea, 8*n_fea, 4, 2, 2),
                SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, 2),
                SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, 2),
            )
            
        self.name = name
    def forward(self, x):
        x1 = self.snconv1(x)
        x2 = self.snconv2(x1)
        x3 = self.snconv3(x2)
        x4 = self.snconv4(x3)
        x = self.snconv5(x4)
        x = x.view((x.size(0),-1))
        fea = torch.cat((x1.view((x1.size(0),-1)),
                        x2.view((x2.size(0),-1)),
                        x3.view((x3.size(0),-1)),
                        x4.view((x4.size(0),-1))), dim=1)
        return x, fea
    
    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_best.h5')
        torch.save(self.state_dict(), filename)
