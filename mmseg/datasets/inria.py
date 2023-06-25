# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class InriaDataset(BaseSegDataset):
    """Inria dataset.

    In segmentation map annotation for Inria, 0 stands for background, 1 stands for 'Buildings'
    ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    METAINFO = dict(
        classes=('background', 'buildings'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self, **kwargs):
        super(InriaDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
