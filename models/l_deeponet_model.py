from torch import nn
import torch
from timm.layers import trunc_normal_
from core import act_list


class L_DeepONet_Model_AE_DON(nn.Module):
    def __init__(self, input_shape, p, latent_dim, q, branch_channels, trunk_layers, AE_layers,
                 stride=1, kernel_size=3, act_norm="Batch", AE_actfun=nn.GELU(),
                 branch_actfun=nn.Mish(), trunk_actfun=nn.Mish(), **kwargs):
        super(L_DeepONet_Model_AE_DON, self).__init__()
        self.m = latent_dim
        self.q = q
        self.H, self.W = input_shape
        self.branch = Branch(p, latent_dim, branch_channels, stride, 
                             kernel_size, act_norm,
                             actfun = act_list[branch_actfun.lower()])
        self.trunk = Trunk(p, latent_dim, trunk_layers,
                           actfun = act_list[trunk_actfun.lower()])
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.H * self.W, AE_layers[0]), AE_actfun,
            *[nn.Sequential(nn.Linear(AE_layers[i], AE_layers[i+1]), AE_actfun)
              for i in range(len(AE_layers)-1)]
        )
        
        self.decoder = nn.Sequential(
            *[nn.Sequential(nn.Linear(AE_layers[-i], AE_layers[-i-1]), AE_actfun)
              for i in range(1, len(AE_layers))],
            nn.Linear(AE_layers[0], self.H * self.W),
            #nn.Sigmoid(),
            nn.Unflatten(1, (self.H, self.W))
        )

    def forward(self, x0, x1, **kwargs):
        nt, _ = x1.size()
        encoded = self.encoder(x0) # (B, latent_dim)
        x0 = encoded.reshape(-1, 1, self.q, self.q) #x0:  # (B, 1, sqrt(latent_dim), sqrt(latent_dim))
        y_branch = self.branch(x0) # (B, latent_dim, p)
        y_trunk = self.trunk(x1)   # (nt, latent_dim, p)
        don = torch.einsum('ijk,pjk->ipj', y_branch, y_trunk) # (B, nt, latent_dim)
        don = don.reshape(-1, self.m)  # (B * nt, latent_dim)
        decoded = self.decoder(don)
        out = decoded.reshape(-1, nt, self.H, self.W)
        return out


class L_DeepONet_Model_EN_DON(nn.Module):
    def __init__(self, input_shape, p, latent_dim, q, branch_channels, trunk_layers, EN_layers,
                 stride=1, kernel_size=3, act_norm="Batch", AE_actfun=nn.GELU(),
                 branch_actfun=nn.Mish(), trunk_actfun=nn.Mish(), **kwargs):
        super(L_DeepONet_Model_EN_DON, self).__init__()
        self.m = latent_dim
        self.q = q
        self.H, self.W = input_shape
        self.branch = Branch(p, latent_dim, branch_channels, stride, 
                             kernel_size, act_norm,
                             actfun = act_list[branch_actfun.lower()])
        self.trunk = Trunk(p, latent_dim, trunk_layers,
                           actfun = act_list[trunk_actfun.lower()])
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.H * self.W, EN_layers[0]), AE_actfun,
            *[nn.Sequential(nn.Linear(EN_layers[i], EN_layers[i+1]), AE_actfun)
              for i in range(len(EN_layers)-1)]
        )


    def forward(self, x0, x1, **kwargs):
        encoded = self.encoder(x0) # (B, latent_dim)
        x0 = encoded.reshape(-1, 1, self.q, self.q) #x0:  # (B, 1, sqrt(latent_dim), sqrt(latent_dim))
        y_branch = self.branch(x0) # (B, latent_dim, p)
        y_trunk = self.trunk(x1)   # (nt, latent_dim, p)
        out = torch.einsum('ijk,pjk->ipj', y_branch, y_trunk) # (B, nt, latent_dim)
        return out


class L_DeepONet_Model_DON(nn.Module):
    def __init__(self, p, latent_dim, branch_channels, trunk_layers, 
                 stride=1, kernel_size=3, act_norm="Batch", 
                 branch_actfun=nn.Mish(), trunk_actfun=nn.Mish(), **kwargs):
        super(L_DeepONet_Model_DON, self).__init__()
        self.m = latent_dim
        self.branch = Branch(p, latent_dim, branch_channels, stride, 
                             kernel_size, act_norm,
                             actfun = act_list[branch_actfun.lower()])
        self.trunk = Trunk(p, latent_dim, trunk_layers,
                           actfun = act_list[trunk_actfun.lower()])

    def forward(self, x0, x1, **kwargs):
        #x0:  # (B, 1, sqrt(latent_dim), sqrt(latent_dim))
        y_branch = self.branch(x0) # (B, latent_dim, p)
        y_trunk = self.trunk(x1)   # (nt, latent_dim, p)
        Y = torch.einsum('ijk,pjk->ipj', y_branch, y_trunk) # (B, nt, latent_dim)
        return Y


class L_DeepONet_Model_AE(nn.Module):
    def __init__(self, input_shape, AE_layers, **kwargs):
        super(L_DeepONet_Model_AE, self).__init__()
        H, W = input_shape
        self.actfun = nn.GELU()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(H * W, AE_layers[0]), self.actfun,
            *[nn.Sequential(nn.Linear(AE_layers[i], AE_layers[i+1]), self.actfun)
              for i in range(len(AE_layers)-1)]
        )
        
        self.decoder = nn.Sequential(
            *[nn.Sequential(nn.Linear(AE_layers[-i], AE_layers[-i-1]), self.actfun)
              for i in range(1, len(AE_layers))],
            nn.Linear(AE_layers[0], H * W),
            #nn.Sigmoid(),
            nn.Unflatten(1, (H, W))
        )

    def forward(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Branch(nn.Module):
    """Branch"""

    def __init__(self, p, latent_dim, channels, stride=1, 
                 kernel_size=3, act_norm="Batch",
                 actfun=nn.Mish()):
        super(Branch, self).__init__()
        self.channel_len = len(channels)
        padding = (kernel_size - stride + 1) // 2
        self.branch = nn.Sequential(
                BasicConv2d(1, channels[0], kernel_size, 
                            stride, padding, act_norm=act_norm, act_fun=actfun),
                *[BasicConv2d(channels[i], channels[i+1], kernel_size, 
                            stride, padding, act_norm=act_norm, act_fun=actfun)
                    for i in range(len(channels)-1)],
                nn.Flatten(),
                nn.Linear(channels[-1] * latent_dim, p * latent_dim),
                nn.Unflatten(1, (latent_dim, p))
        )

    def forward(self, x): # (B, 1, sqrt(latent_dim), sqrt(latent_dim))
        x = self.branch(x)
        return x    # (B, latent_dim, p)


class Trunk(nn.Module):
    """Trunk"""

    def __init__(self, p, latent_dim, layers,
                 actfun=nn.Mish()):
        super(Trunk, self).__init__()
        self.trunk = nn.Sequential(
                nn.Linear(1, layers[0]), actfun,
                *[nn.Sequential(nn.Linear(layers[i], layers[i+1]), actfun)
                for i in range(len(layers)-1)],
                nn.Linear(layers[-1], p * latent_dim),
                nn.Unflatten(1, (latent_dim, p))
        )

    def forward(self, x): # (nt, 1)
        x = self.trunk(x)
        return x    # (nt, latent_dim, p)


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 act_norm="Batch",
                 act_fun = nn.Mish()):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)
        if act_norm == "Batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif act_norm == "Group":
            self.norm = nn.GroupNorm(2, out_channels)
        
        self.act = act_fun

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm =="Batch":
            y = self.norm(self.act(y))
        elif self.act_norm =="Group":
            y = self.act(self.norm(y))
        else:
            y = self.act(y)
        return y

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)