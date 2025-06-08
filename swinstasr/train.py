import os.path as osp
import swinstasr.archs
import swinstasr.data
import swinstasr.models
import swinstasr.losses
import swinstasr.metrics
from basicsr.train import train_pipeline

import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
