from mmseg.models.builder import HEADS
from .ws_head import WS

@HEADS.register_module()
class LAWINHeadWS(WS):
    def __init__(self, depth, concat_fuse, decoder_params, **kwargs):
        super(LAWINHeadWS, self).__init__(**kwargs)