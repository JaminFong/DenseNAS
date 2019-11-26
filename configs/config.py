from tools.collections import AttrDict

__C = AttrDict()

cfg = __C

__C.net_config=""

__C.train_params=AttrDict()

__C.train_params.batch_size=256
__C.train_params.num_workers=8


__C.optim=AttrDict()

__C.optim.init_dim=16
__C.optim.bn_momentum=0.1
__C.optim.bn_eps=0.001


__C.data=AttrDict()

__C.data.dataset='imagenet'
__C.data.train_data_type='lmdb'
__C.data.val_data_type='img'
__C.data.patch_dataset=False
__C.data.num_examples=1281167
__C.data.input_size=(3,224,224)