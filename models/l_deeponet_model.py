from torch import nn
import torch
from core import act_list
from modules import ConvSC, DeepConv2d

class L_DeepONet_Model_AE_DON(nn.Module):
    def __init__(self, input_shape, p, latent_dim, q, branch_channels, trunk_layers, AE_layers,
                 stride=1, kernel_size=3, act_norm="Batch", AE_actfun="gelu",
                 branch_actfun="gelu", trunk_actfun="gelu", **kwargs):
        super(L_DeepONet_Model_AE_DON, self).__init__()
        self.m = latent_dim
        self.q = q
        self.H, self.W = input_shape
        self.branch = Branch(p, latent_dim, branch_channels, stride, 
                             kernel_size, act_norm,
                             actfun = act_list[branch_actfun.lower()])
        self.trunk = Trunk(p, latent_dim, trunk_layers,
                           actfun = act_list[trunk_actfun.lower()])
        
        self.encoder = Encoder_Linear(self.H, self.W, AE_layers, act_list[AE_actfun.lower()])
        self.decoder = Decoder_Linear(self.H, self.W, AE_layers, act_list[AE_actfun.lower()])

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
                 stride=1, kernel_size=3, act_norm="Batch", AE_actfun="gelu",
                 branch_actfun="gelu", trunk_actfun="gelu", **kwargs):
        super(L_DeepONet_Model_EN_DON, self).__init__()
        self.m = latent_dim
        self.q = q
        H, W = input_shape
        self.branch = Branch(p, latent_dim, branch_channels, stride, 
                             kernel_size, act_norm,
                             actfun = act_list[branch_actfun.lower()])
        self.trunk = Trunk(p, latent_dim, trunk_layers,
                           actfun = act_list[trunk_actfun.lower()])
        
        self.encoder = Encoder_Linear(H, W, EN_layers, act_list[AE_actfun.lower()])


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
                 branch_actfun="gelu", trunk_actfun="gelu", **kwargs):
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
    def __init__(self, input_shape, AE_layers, actfun="gelu", **kwargs):
        super(L_DeepONet_Model_AE, self).__init__()
        H, W = input_shape
        self.encoder = Encoder_Linear(H, W, AE_layers, act_list[actfun.lower()])
        self.decoder = Decoder_Linear(H, W, AE_layers, act_list[actfun.lower()])

    def forward(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class L_DeepONet_Model_AE_Conv(nn.Module):
    def __init__(self, input_shape, AE_channels, spatio_kernel=3, 
                if_mid=True, act_inplace = False,**kwargs):
        super(L_DeepONet_Model_AE_Conv, self).__init__()
        H, W = input_shape
        self.encoder = Encoder_Conv(AE_channels, spatio_kernel,if_mid,act_inplace)
        
        self.decoder = Decoder_Conv(AE_channels, spatio_kernel,if_mid,act_inplace)

    def forward(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def sampling_generator(N, reverse=False, if_mid=True):
    samplings_batch = [False, True] if if_mid else [True, True]
    samplings = samplings_batch * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder_Conv(nn.Module):
    """Encoder of Conv AE"""

    def __init__(self, AE_channels, spatio_kernel, if_mid=True, act_inplace=False):
        N_E = len(AE_channels)
        samplings = sampling_generator(N_E, if_mid=if_mid)
        super(Encoder_Conv, self).__init__()
        self.encoder = nn.Sequential(
              ConvSC(1, AE_channels[0], spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(AE_channels[i], AE_channels[i+1], spatio_kernel, downsampling=samplings[i+1],
                     act_inplace=act_inplace) for i in range(N_E-1)]
        )

    def forward(self, x):  
        latent = self.encoder(x)
        return latent


class Decoder_Conv(nn.Module):
    """DEcoder of Conv AE"""

    def __init__(self, AE_channels, spatio_kernel, if_mid=True, act_inplace=False):
        N_D = len(AE_channels)
        samplings = sampling_generator(N_D, reverse=True, if_mid=if_mid)
        super(Decoder_Conv, self).__init__()
        self.decoder = nn.Sequential(
            *[ConvSC(AE_channels[-i], AE_channels[-i-1], spatio_kernel, upsampling=samplings[i-1],
                     act_inplace=act_inplace) for i in range(1, N_D)],
              ConvSC(AE_channels[0], 1, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace, act_norm=False)
        )

    def forward(self, latent):
        Y = self.decoder(latent)
        return Y


class Encoder_Linear(nn.Module):
    """Encoder of Linear AE"""

    def __init__(self, H:int, W:int, AE_layers:list, actfun=nn.gelu()):
        super(Encoder_Linear, self).__init__()
        N_E = len(AE_layers)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(H * W, AE_layers[0]), actfun,
            *[nn.Sequential(nn.Linear(AE_layers[i], AE_layers[i+1]), actfun)
              for i in range(N_E-1)]
        )

    def forward(self, x): # (B, H, W)
        x = self.encoder(x)
        return x    # (B, latent_dim)


class Decoder_Linear(nn.Module):
    """Decoder of Linear AE"""

    def __init__(self, H:int, W:int, AE_layers:list, actfun=nn.gelu()):
        super(Decoder_Linear, self).__init__()
        N_D = len(AE_layers)
        self.decoder = nn.Sequential(
            *[nn.Sequential(nn.Linear(AE_layers[-i], AE_layers[-i-1]), actfun)
              for i in range(1, N_D)],
            nn.Linear(AE_layers[0], H * W),
            nn.Unflatten(1, (1, H, W))
        )

    def forward(self, x): # (B, latent_dim)
        x = self.decoder(x)
        return x    # (B, H, W)



class Branch(nn.Module):
    """Branch"""

    def __init__(self, p:int, latent_dim:int, channels:list, stride=1, 
                 kernel_size=3, act_norm="Batch",
                 actfun=nn.gelu()):
        super(Branch, self).__init__()
        self.channel_len = len(channels)
        padding = (kernel_size - stride + 1) // 2
        self.branch = nn.Sequential(
                DeepConv2d(1, channels[0], kernel_size, 
                            stride, padding, act_norm=act_norm, act_fun=actfun),
                *[DeepConv2d(channels[i], channels[i+1], kernel_size, 
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

    def __init__(self, p:int, latent_dim:int, layers:list,
                 actfun=nn.gelu()):
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
