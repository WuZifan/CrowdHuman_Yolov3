import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr


def parse_module_defs(module_defs):
    '''

    :param module_defs: [{'type': 'convolutional', 'batch_normalize': '1', 'filters': '32', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}。。。。。]
    :return: CBL_idx：[1,2,3,4] 都是id,表示用了BN层的block——id
             Conv_idx: [1,2,3,4] 都是id，表示没用BN层的block-id
             prune_idx：[1,2,3,4] 都是id，最后确定需要被剪枝的block-id
    '''

    CBL_idx = []
    Conv_idx = []
    '''
        统计哪些卷积层用了BN，哪些没用
    '''
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)


    '''
        这里可以理解为每一个resblock中，非shortcut那一路中
        最后一个conv不做prune
        以及shortcut，from的的最近的那一个conv不做
        这个理解没错的，实际验证了一下就是这样的
    '''
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)

    ignore_idx.add(84)
    ignore_idx.add(96)

    '''
        仅考虑不在shortBlock中的，且含BN层的卷积。
    '''
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx


def gather_bn_weights(module_list, prune_idx):
    '''

    :param module_list:
    :param prune_idx:
    :return:
    '''

    '''
        module_list[ids] 拿到这个list中第几个sequential
        module_lust[ids][1] 拿到这个sequential中第1个元素，一般是BN层
        然后就拿到这个BN层的数据的尺寸，就和它对接的CNN层的filters一样。

    '''
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def write_cfg(cfg_file, module_defs):

    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write("[{}]\n".format(module_def['type']))
            for key, value in module_def.items():
                if key != 'type':
                    f.write("{}={}\n".format(key,value))
            f.write("\n")
    return cfg_file


class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def obtain_quantiles(bn_weights, num_quantile=5):

    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        ['{}/{}'.format(i,num_quantile) for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)

    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))
        if len(route_in_idxs) == 1:
            return CBLidx2mask[route_in_idxs[0]]
        elif len(route_in_idxs) == 2:
            return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
        else:
            print("Something wrong with route module!")
            raise Exception


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):

    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()


def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):
    '''

    :param model: 模型
    :param prune_idx: 需要被prune的block id
    :param CBL_idx:  含有BN层的block的id
    :param CBLidx2mask: blockid和mask的dict
    :return:
    '''

    pruned_model = deepcopy(model)
    # 对于每一个需要被prune的block
    for idx in prune_idx:
        # 拿到mask
        if torch.cuda.is_available():
            mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        else:
            mask = torch.from_numpy(CBLidx2mask[idx])
        # 拿到这一个block层对应的BN层
        bn_module = pruned_model.module_list[idx][1]

        # bn层参数*mask，进行剪枝
        bn_module.weight.data.mul_(mask)

        '''
            保留被剪枝的BN层的bias？
        '''
        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
        print('activation',activation.shape)

        # 两个上采样层前的卷积层
        next_idx_list = [idx + 1]
        if idx == 79:
            next_idx_list.append(84)
        elif idx == 91:
            next_idx_list.append(96)

        '''
        因为BN=gamma*α+bias，而mask仅处理了gamma的内容，还要考虑bias产生的影响

        1、如果下一层不是卷积层，那么希望保留bias的影响，因此需要在下一层的bias中，
           加上这一个BLOCK中被prune的BN层的bias参数产生的激活值
        2、如果下一层是卷积层，那么我也会影响你的BN层的均值，因此就在你的BN层中减去
           我的产生的均值，让我对你基本没有影响？

        为什么下一层是卷积层和不是卷积层，两者的处理方式不一样呢？？
        '''
        for next_idx in next_idx_list:
            # 那个下一个block的第0个模块
            next_conv = pruned_model.module_list[next_idx][0]
            # 算一下下一个block的卷积木块的feature_map的h+w？
            print(next_conv.weight.data.shape)
            # 把每一个卷积核的都加起来了
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            print(conv_sum.shape)
            print(conv_sum)
            # import time
            # time.sleep(1000)
            # 这个相当于把prune产生影响的输出，和下一层的CNN层一起，做一次卷积
            # 从而来计算prune对下一层产生的偏移量？
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            if next_idx in CBL_idx:
                # 如果下一层是含有BN层的block，那么在这个block中减去这个offset
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
            else:
                # 如果下一层不是卷积block，那么在这个block中加上这个offset？
                next_conv.bias.data.add_(offset)

        bn_module.bias.data.mul_(mask)

    return pruned_model


def obtain_bn_mask(bn_module, thre):
    '''
    判断这个BN模块中的参数和thre的大小关系。
    最后输出用01表示的mask，1表示大于，0表示小于。
    ge()就是用来判断大小的。
    :param bn_module: 某一个BN木块
    :param thre: 一个标量
    :return:
    '''

    if torch.cuda.is_available():
        thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask
