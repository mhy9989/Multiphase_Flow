from torch import nn
from neuralop.models import FNO
from modules import ConvSC


class FNO_Model(nn.Module):
    def __init__(self, data_shape, n_modes, n_layers, hid, in_ch, out_ch, 
                 spatio_kernel_enc=3, spatio_kernel_dec=3, N_S=4, **kwargs):
        super(FNO_Model, self).__init__()
        H, W = data_shape
        act_inplace = False
        self.enc = Encoder(in_ch, hid, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid, out_ch, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        
        self.operator = nn.Sequential(
                *[FNO(n_modes=n_modes, hidden_channels=hid,
                    in_channels=hid, out_channels=hid) for _ in range(n_layers)],
                )

    def forward(self, x, **kwargs):
        hid, skip = self.enc(x)
        for i in range(len(self.operator)):
            hid = self.operator[i](hid)
        Y = self.dec(hid, skip)
        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]



class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y
