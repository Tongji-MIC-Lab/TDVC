from yacs.config import CfgNode
_C = CfgNode()

_C.batch_size = 1
_C.workers = 0
_C.compress = True
_C.qp = 22
_C.resume = ""
_C.output_dir = ""
_C.clip = 0
_C.amp = False
_C.lr = 0.01
_C.dataset_path = ""
_C.compress_path = ""
_C.lambda_ = 0


def get_cfg():
    return _C.clone()
