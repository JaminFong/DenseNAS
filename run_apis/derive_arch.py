import logging


class BaseArchGenerate(object):
    def __init__(self, super_network, config):
        self.config = config
        self.num_blocks = len(super_network.block_chs)  # including the input 32 and 1280 block
        self.super_chs = super_network.block_chs
        self.input_configs = super_network.input_configs


    def update_arch_params(self, betas, head_alphas, stack_alphas):
        self.betas = betas
        self.head_alphas = head_alphas
        self.stack_alphas = stack_alphas


    def derive_chs(self):
        """
        using viterbi algorithm to choose the best path of the super net
        """
        path_p_max = []  # [[max_last_state_id, trans_prob], ...]

        path_p_max.append([0, 1])

        for input_config in self.input_configs:
            block_path_prob_max = [None, 0]
            for in_block_id, beta_id in zip(input_config['in_block_idx'], input_config['beta_idx']):
                path_prob = path_p_max[in_block_id][1]*self.betas[in_block_id][beta_id]
                if path_prob > block_path_prob_max[1]:
                    block_path_prob_max = [in_block_id ,path_prob]

            path_p_max.append(block_path_prob_max)
        
        ch_idx = len(path_p_max) - 1
        ch_path = []
        ch_path.append(ch_idx)

        while 1:
            ch_idx = path_p_max[ch_idx][0]
            ch_path.append(ch_idx) 
            if ch_idx == 0:
                break

        derived_chs = [self.super_chs[ch_id] for ch_id in ch_path]

        ch_path = ch_path[::-1]
        derived_chs = derived_chs[::-1]

        return ch_path, derived_chs

    
    def derive_ops(self, alpha, alpha_type):
        assert alpha_type in ['head', 'stack']

        if alpha_type == 'head':
            op_type = self.config.search_params.PRIMITIVES_head[alpha.index(max(alpha))]
        elif alpha_type == 'stack':
            op_type = self.config.search_params.PRIMITIVES_stack[alpha.index(max(alpha))]

        return op_type
        
    
    def derive_archs(self, betas, head_alphas, stack_alphas, if_display=True):
        raise NotImplementedError