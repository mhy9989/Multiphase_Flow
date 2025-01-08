from torch import nn

class DeepONet_Model(nn.Module):
    def __init__(self, in_shape, branch_list, trunk_list):
      super(DeepONet_Model, self).__init__()
      H, W = in_shape
      self.actfun = nn.SiLU()
      self.branch = nn.Sequential(
            *[[nn.Linear(branch_list[i],branch_list[i+1]), self.actfun]
               for i in range(len(branch_list)-2)],
               nn.Linear(branch_list[-2], branch_list[-1])
               )
      self.trunk = nn.Sequential(
            *[[nn.Linear(trunk_list[i],trunk_list[i+1]), self.actfun]
               for i in range(len(trunk_list)-2)],
               nn.Linear(trunk_list[-2], trunk_list[-1])
               )

    def forward(self, x1, x2, **kwargs):
      for i in range(len(self.branch)):
            x1 = self.branch[i](x1)
      for i in range(len(self.trunk)):
            x2 = self.trunk[i](x2)
      y = x1 * x2
      return y
