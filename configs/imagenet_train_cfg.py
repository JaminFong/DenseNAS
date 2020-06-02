from tools.collections import AttrDict

__C = AttrDict()

cfg = __C

__C.net_type='mbv2' # mbv2 / res
__C.net_config="""[[16, 16], 'mbconv_k3_t1', [], 0, 1]|
[[16, 24], 'mbconv_k3_t6', [], 0, 2]|
[[24, 48], 'mbconv_k7_t6', ['mbconv_k3_t3'], 1, 2]|
[[48, 72], 'mbconv_k5_t6', ['mbconv_k3_t6', 'mbconv_k3_t3'], 2, 2]|
[[72, 128], 'mbconv_k3_t6', ['mbconv_k3_t3', 'mbconv_k3_t3'], 2, 1]|
[[128, 160], 'mbconv_k3_t6', ['mbconv_k7_t3', 'mbconv_k5_t6', 'mbconv_k7_t3'], 3, 2]|
[[160, 176], 'mbconv_k3_t3', ['mbconv_k3_t6', 'mbconv_k3_t6', 'mbconv_k3_t6'], 3, 1]|
[[176, 384], 'mbconv_k7_t6', [], 0, 1]|
[[384, 1984], 'conv1_1']"""

__C.train_params=AttrDict()
__C.train_params.epochs=240
__C.train_params.use_seed=True
__C.train_params.seed=0

__C.optim=AttrDict()
__C.optim.init_lr=0.5
__C.optim.min_lr=1e-5
__C.optim.lr_schedule='cosine'  # cosine poly
__C.optim.momentum=0.9
__C.optim.weight_decay=4e-5
__C.optim.use_grad_clip=False
__C.optim.grad_clip=10
__C.optim.label_smooth=True
__C.optim.smooth_alpha=0.1
__C.optim.init_mode='he_fout'

__C.optim.if_resume=False
__C.optim.resume=AttrDict()
__C.optim.resume.load_path=''
__C.optim.resume.load_epoch=0

__C.optim.use_warm_up=False
__C.optim.warm_up=AttrDict()
__C.optim.warm_up.epoch=5
__C.optim.warm_up.init_lr=0.0001
__C.optim.warm_up.target_lr=0.1

__C.optim.use_multi_stage=False
__C.optim.multi_stage=AttrDict()
__C.optim.multi_stage.stage_epochs=330

__C.optim.cosine=AttrDict()
__C.optim.cosine.use_restart=False
__C.optim.cosine.restart=AttrDict()
__C.optim.cosine.restart.lr_period=[10, 20, 40, 80, 160, 320]
__C.optim.cosine.restart.lr_step=[0, 10, 30, 70, 150, 310]

__C.optim.bn_momentum=0.1
__C.optim.bn_eps=0.001

__C.data=AttrDict()
__C.data.num_workers=16
__C.data.batch_size=1024
__C.data.dataset='imagenet' #imagenet
__C.data.train_data_type='lmdb'
__C.data.val_data_type='img'
__C.data.patch_dataset=False
# __C.data.num_examples=1281167
__C.data.input_size=(3,224,224)
__C.data.type_of_data_aug='random_sized'  # random_sized / rand_scale
__C.data.random_sized=AttrDict()
__C.data.random_sized.min_scale=0.08
__C.data.mean=[0.485, 0.456, 0.406]
__C.data.std=[0.229, 0.224, 0.225]
__C.data.color=False