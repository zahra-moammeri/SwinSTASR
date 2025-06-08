import os.path as osp
import swinstasr.archs
import swinstasr.data
import swinstasr.models
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
