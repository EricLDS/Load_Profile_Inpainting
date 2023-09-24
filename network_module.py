import torch
from torch import nn
from torch.nn import functional as F


class GatedConv1dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv1d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv1dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm1d = torch.nn.BatchNorm1d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                
    def gated(self, mask):
        return self.sigmoid(mask)
    
    def forward(self, input):
        raw = self.conv1d(input)
        mask = self.mask_conv1d(input)
        
        
        if self.activation is not None:
            x = self.activation(raw) * self.gated(mask)
        else:
            x = raw * self.gated(mask)
            
        if self.batch_norm:
            return self.batch_norm1d(x), raw, self.gated(mask)
        else:
            return x, raw, self.gated(mask)

class GatedDeConv1dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv1d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv1dWithActivation, self).__init__()
        self.gcovd1d = GatedConv1dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        x, raw, score = self.gcovd1d(x)
        return x, raw, score


class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv1d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

