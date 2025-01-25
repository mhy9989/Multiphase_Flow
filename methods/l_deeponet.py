from .sao import SAO
from models import (L_DeepONet_Model_DON, L_DeepONet_Model_AE, 
                    L_DeepONet_Model_AE_DON, L_DeepONet_Model_EN_DON,
                    L_DeepONet_Model_AE_Conv)
import numpy as np
import torch

class L_DeepONet(SAO):

    def __init__(self, args, ds_config, base_criterion):
        SAO.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(args)
        self.model1 = None
        self.m = args.latent_dim
        self.args = args
        self.torch_scaler = None
        X_loc = np.array([i*0.1 for i in args.data_after]).reshape(args.data_after_num, 1)
        self.x1 = torch.Tensor((X_loc-X_loc.mean())/X_loc.std()).to(self.device) if args.data_after_num != 1 else torch.Tensor(X_loc).to(self.device)

    def build_model(self, args):
        if args.model_type == "AE":
            return L_DeepONet_Model_AE(**args).to(self.device)
        elif args.model_type == "AE_Conv":
            return L_DeepONet_Model_AE_Conv(**args).to(self.device)
        elif args.model_type == "DON":
            return L_DeepONet_Model_DON(**args).to(self.device)
        elif args.model_type == "AE_DON":
            return L_DeepONet_Model_AE_DON(**args).to(self.device)
        elif args.model_type == "EN_DON":
            return L_DeepONet_Model_EN_DON(**args).to(self.device)
    
    
    def predict_AE_en(self, batch_x):
        """Forward the AE en model"""
        pred_y = self.model.encoder(batch_x)
        return pred_y
    
    def predict_AE_de(self, batch_x):
        """Forward the AE de model"""
        pred_y = self.model.decoder(batch_x)
        return pred_y
    
    def predict(self, batch_x):
        """Forward the model"""
        if self.args.model_type not in ["AE", "AE_Conv"]:
            pred_y = self.model(batch_x, self.x1)
        else:
            pred_y = self.model(batch_x)
        return pred_y

