
# Copyright (c) Shanghai AI Lab. All rights reserved.
from .upernetws import UPerHeadWS
from .fpn_headws import FPNHeadWS
from .sep_aspp_headws import DepthwiseSeparableASPPHeadWS
from .ham_head import LightHamHead
from .ham_headws import LightHamHeadWS

__all__ = ['UPerHeadWS', 'FPNHeadWS', 'DepthwiseSeparableASPPHeadWS', 'LightHamHead', 'LightHamHeadWS']
