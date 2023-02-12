# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import Upsample, resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from .ws_head import WS

@HEADS.register_module()
class FPNHeadWS(WS):
    def __init__(self, feature_strides, **kwargs):
        super(FPNHeadWS, self).__init__(**kwargs)