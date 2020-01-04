from tools.collections import AttrDict

__C = AttrDict()

cfg = __C

__C.net_config=""

__C.data=AttrDict()
__C.data.num_workers=16
__C.data.batch_size=1024
__C.data.dataset='imagenet'
__C.data.train_data_type='lmdb'
__C.data.val_data_type='lmdb'
__C.data.patch_dataset=False
__C.data.num_examples=1281167
__C.data.input_size=(3,224,224)