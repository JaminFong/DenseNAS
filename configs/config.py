from tools.collections import AttrDict

__C = AttrDict()

cfg = __C

__C.net_config="""[[16, 16], 'mbconv_k3_t1', [], 0, 1]|
[[16, 24], 'mbconv_k5_t3', ['mbconv_k5_t3', 'mbconv_k3_t3'], 2, 2]|
[[24, 48], 'mbconv_k5_t6', [], 0, 2]|
[[48, 80], 'mbconv_k5_t6', ['mbconv_k7_t3', 'mbconv_k5_t3', 'mbconv_k3_t3'], 3, 2]|
[[80, 112], 'mbconv_k3_t3', ['mbconv_k3_t3'], 1, 1]|
[[112, 160], 'mbconv_k7_t6', ['mbconv_k7_t3', 'mbconv_k7_t3'], 2, 2]|
[[160, 352], 'mbconv_k5_t6', ['mbconv_k3_t3'], 1, 1]|
[[352, 416], 'mbconv_k3_t3', [], 0, 1]|
[[416, 480], 'mbconv_k3_t3', [], 0, 1]"""

__C.train_params=AttrDict()

__C.train_params.batch_size=256
__C.train_params.num_workers=8


__C.optim=AttrDict()

__C.optim.last_dim=1728
__C.optim.init_dim=16
__C.optim.bn_momentum=0.1
__C.optim.bn_eps=0.001


__C.data=AttrDict()

__C.data.dataset='imagenet' # cifar10 imagenet
__C.data.train_data_type='lmdb'
__C.data.val_data_type='img'
__C.data.patch_dataset=False
__C.data.num_examples=1281167
__C.data.input_size=(3,224,224)