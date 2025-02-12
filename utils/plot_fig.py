import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import print_log
import os
import os.path as osp
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
config = {
    "font.family":'Times New Roman',
    "font.size": 13,
    "mathtext.fontset": 'stix',
    "mathtext.rm": 'Times New Roman',
    "mathtext.it": 'Times New Roman:italic',
    "mathtext.bf": 'Times New Roman:bold'
}
rcParams.update(config)

def plot_figure(x_mesh, y_mesh, min_max, data, data_name, mode, pic_folder, dpi=300):
    cmap = 'Blues'
    levels = np.linspace(min_max[0], min_max[1], 600)
    data = np.clip(data, min_max[0], min_max[1])
    map = plt.contourf(x_mesh, y_mesh, data, levels,cmap=cmap) 
    pic_name = f'{mode}_{data_name}.png'
    ax = plt.gca()
    ax.set_aspect(1) 
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    plt.colorbar(map,fraction=0.02, pad=0.03,
                    ticks=np.linspace(min_max[0], min_max[1], 5),
                    format = '%.1e')
    plt.title(f"{mode} {data_name} data", fontsize=15)
    
    plt.xlabel('$\mathit{X}$/mm', fontsize=15)
    plt.ylabel('$\mathit{Y}$/mm', fontsize=15)
    pic_path = osp.join(pic_folder, pic_name)
    plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_learning_curve(loss_record, model_path, dpi=300, title='', dir_name = "pic"):
    ''' Plot learning curve of your DNN (train & valid loss) '''
    total_steps = len(loss_record['train_loss'])
    x_1 = range(total_steps)
    plt.semilogy(x_1, loss_record['train_loss'], c='tab:red', label='train')
    if len(loss_record['valid_loss']) != 0:
        x_2 = x_1[::len(loss_record['train_loss']) // len(loss_record['valid_loss'])]
        plt.semilogy(x_2, loss_record['valid_loss'], c='tab:cyan', label='valid')
    plt.xlabel('Training steps', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Learning curve of {}'.format(title), fontsize=15)
    plt.legend()

    pic_name = f'loss_record.png'
    pic_folder = osp.join(model_path, dir_name)
    os.makedirs(pic_folder, exist_ok=True)
    pic_path =osp.join(pic_folder, pic_name)
    print_log(f'Simulation picture saved in {pic_path}')
    plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
    plt.close()
