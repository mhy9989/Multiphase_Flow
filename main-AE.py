from modeltrain import modeltrain
from modelbuild import modelbuild
import os
import argparse
import deepspeed


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
    modelname = 'L_DeepONet-AE'
    ## model path
    dir_path = os.path.dirname(os.path.abspath(__file__))

    ds_args = add_argument()
    model_path = os.path.join(dir_path, 'Model', f'{modelname}')
    total_data = modelbuild(model_path, ds_args)
    model_data = total_data.get_data()
    model = modeltrain(model_data, model_path)
    model.train()
    model.test_inference()


if __name__ == '__main__':
    main()
