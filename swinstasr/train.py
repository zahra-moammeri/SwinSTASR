import os.path as osp
import swinstasr.archs
import swinstasr.data
import swinstasr.models
import swinstasr.losses
import swinstasr.metrics
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
