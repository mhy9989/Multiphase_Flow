import time
import torch
from timm.utils import AverageMeter

from utils import reduce_tensor, get_progress
from .sao import SAO
from models import TFNO_Model
from core.lossfun import Regularization

class TFNO(SAO):

    def __init__(self, args, ds_config, base_criterion):
        SAO.__init__(self, args, ds_config, base_criterion)
        self.args.n_modes = args.n_modes = tuple(args.n_modes)
        self.model = self.build_model(args)

    def build_model(self, args):
        return TFNO_Model(**args).to(self.device)