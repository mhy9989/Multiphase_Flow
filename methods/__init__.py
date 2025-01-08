from .sao import SAO
from .fno import FNO
from .tfno import TFNO
from .msta import MSTA
from .l_deeponet import L_DeepONet


method_maps = {
    'sao': SAO,
    'fno': FNO,
    'tfno': TFNO,
    'msta': MSTA,
    'l-deeponet': L_DeepONet

}

__all__ = [
    'method_maps', 'sao', 'fno', 'tfno', 'msta', 'l-deeponet'
]