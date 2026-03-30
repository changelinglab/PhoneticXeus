# TODO(shikhar): this is a temporary hack to make xeus work with espnet3 and espnet2
import sys

try:
    # shikhar
    import espnet2.legacy as _espnet

    ESPNET_VERSION = "espnet2"
except ImportError:
    # yoonjae
    import espnet as _espnet

    ESPNET_VERSION = "espnet1"

sys.modules["espnet_import"] = _espnet
