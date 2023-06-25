# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FLAIRDataset(BaseSegDataset):
    """FLAIR dataset.

    The ``img_suffix`` and ``seg_map_suffix`` are both fixed to '.png'.
    The labels with value 255 are ignored.
    """
    METAINFO = dict(
        classes=('building', 'pervious surface', 'impervious surface', 'bare soil', 'water', 'coniferous', 'deciduous',
                 'brushwood', 'vineyard', 'herbaceous vegetation', 'agricultural land', 'plowed land'),
        palette=[[219, 14, 154], [147, 142, 123], [248, 12, 0], [169, 113, 1], [21, 83, 174], [25, 74, 38],
                 [70, 228, 131], [243, 166, 13], [102, 0, 130], [85, 255, 0], [255, 243, 13], [228, 223, 124]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
