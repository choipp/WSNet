# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM
from .ws_head import WS
@HEADS.register_module()
class UPerHeadWS(WS):
    def __init__(self, pool_scales, **kwargs):
        super(UPerHeadWS, self).__init__(**kwargs)