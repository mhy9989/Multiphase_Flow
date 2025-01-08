from .sao import SAO
from models import SimVP_Model

class MSTA(SAO):

    def __init__(self, args, ds_config, base_criterion):
        SAO.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(args)

    def build_model(self, args):
        return SimVP_Model(**args).to(self.device)


