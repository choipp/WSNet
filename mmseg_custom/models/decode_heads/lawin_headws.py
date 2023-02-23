from mmseg.models.builder import HEADS
from .ws_head import WS

@HEADS.register_module()
class LAWINHeadWS(WS):
    def __init__(self, **kwargs):
        super(LAWINHeadWS, self).__init__(**kwargs)