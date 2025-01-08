from modeltrain import modeltrain
from modelbuild import modelbuild
import os
import argparse
import deepspeed
import os.path as osp
import torch

def add_argument():
    parser = argparse.ArgumentParser(description='CFD-CNN')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main():
    ## model name
    modelname0 = 'L_DeepONet-EN_DON-gelu'
    modelname1 = 'L_DeepONet-AE'
    ## model path
    dir_path = os.path.dirname(os.path.abspath(__file__))

    ds_args = add_argument()
    model_path = os.path.join(dir_path, 'Model', f'{modelname0}')
    total_data = modelbuild(model_path, ds_args)
    model_data = total_data.get_data()
    model = modeltrain(model_data, model_path)
    model.train()

    model_path1 = os.path.join(dir_path, 'Model', f'{modelname1}')
    total_data1 = modelbuild(model_path1, ds_args, init=False)
    model_data1 = total_data1.get_data()
    model1 = modeltrain(model_data1, model_path1)

    best_model_path = osp.join(model1.checkpoints_path, 'checkpoint.pth')
    model1.load_from_state_dict((torch.load(best_model_path)))
    model.method1 = model1.method
    model.test_inference_n()




if __name__ == '__main__':
    main()
