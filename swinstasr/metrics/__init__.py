import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with '_loss.py'
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder)]
# import all the loss modules
_model_modules = [importlib.import_module(f'swinstasr.metrics.{file_name}') for file_name in metric_filenames]