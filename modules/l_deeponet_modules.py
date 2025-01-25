from torch import nn
from timm.layers  import trunc_normal_

class DeepConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 act_norm="Batch",
                 act_fun = nn.Mish()):
        super(DeepConv2d, self).__init__()
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