
# Copyright (c) Shanghai AI Lab. All rights reserved.
from .ws_head import WS
from .upernet_headws import UPerHeadWS
from .fpn_headws import FPNHeadWS
from .sep_aspp_headws import DepthwiseSeparableASPPHeadWS
from .ham_head import LightHamHead
from .ham_headws import LightHamHeadWS
from .lawin_head import LAWINHead
from .lawin_headws import LAWINHeadWS
from .psp_headws import PSPHeadWS

__all__ = ['WS', 'UPerHeadWS', 'FPNHeadWS', 'DepthwiseSeparableASPPHeadWS', 'LightHamHead', 'LightHamHeadWS',
           'LAWINHead', 'LAWINHeadWS', 'PSPHeadWS' ]
