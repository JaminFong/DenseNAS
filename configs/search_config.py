from tools.collections import AttrDict

__C = AttrDict()
search_cfg = __C


__C.search_params=AttrDict()
__C.search_params.arch_update_epoch=10
__C.search_params.val_start_epoch=120
__C.search_params.sample_policy='prob'  # prob uniform
__C.search_params.weight_sample_num=1
__C.search_params.softmax_temp=1.

__C.search_params.adjoin_connect_nums = []
__C.search_params.net_scale = AttrDict()
__C.search_params.net_scale.chs = []
__C.search_params.net_scale.fm_sizes = []
__C.search_params.net_scale.stage = []
__C.search_params.net_scale.num_layers = []

__C.search_params.PRIMITIVES_stack = [
                                        'mbconv_k3_t3',
                                        'mbconv_k3_t6',
                                        'mbconv_k5_t3',
                                        'mbconv_k5_t6',
                                        'mbconv_k7_t3',
                                        'mbconv_k7_t6',
                                        'skip_connect',
                                    ]
__C.search_params.PRIMITIVES_head = [
                                    'mbconv_k3_t3',
                                    'mbconv_k3_t6',
                                    'mbconv_k5_t3',
                                    'mbconv_k5_t6',
                                    'mbconv_k7_t3',
                                    'mbconv_k7_t6',
                                    ]

__C.optim=AttrDict()
__C.optim.init_dim=16
__C.optim.head_dim=16
__C.optim.last_dim=1984
__C.optim.weight=AttrDict()
__C.optim.weight.init_lr=0.1
__C.optim.weight.min_lr=1e-4
__C.optim.weight.lr_decay_type='cosine'
__C.optim.weight.momentum=0.9
__C.optim.weight.weight_decay=4e-5

__C.optim.arch=AttrDict()
__C.optim.arch.alpha_lr=3e-4
__C.optim.arch.beta_lr=3e-4
__C.optim.arch.weight_decay=1e-3

__C.optim.if_sub_obj=True
__C.optim.sub_obj=AttrDict()
__C.optim.sub_obj.type='latency' # latency / flops
__C.optim.sub_obj.skip_reg=False
__C.optim.sub_obj.log_base=15.0
__C.optim.sub_obj.sub_loss_factor=0.1
__C.optim.sub_obj.latency_list_path=''

