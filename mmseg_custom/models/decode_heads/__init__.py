
# Copyright (c) Shanghai AI Lab. All rights reserved.
from .fpn_headws import FPNHeadWS
from .sep_aspp_headws import DepthwiseSeparableASPPHeadWS
from .ham_head import LightHamHead
from .ham_headws import LightHamHeadWS
from .lawin_head import LAWINHead
from .lawin_headws import LAWINHeadWS
from .psp_headws import PSPHeadWS
from .ws_head import WS
__all__ = ['FPNHeadWS', 'DepthwiseSeparableASPPHeadWS', 'LightHamHead', 'LightHamHeadWS',
           'LAWINHead', 'LAWINHeadWS', 'PSPHeadWS' ]
