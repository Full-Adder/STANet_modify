import torch.nn as nn

__all__ = ['remove_layer', 'replace_layer', 'initialize_weights']


def remove_layer(state_dict, keyword):             # 删除模型参数字典的键值对（字典，键）
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def replace_layer(state_dict, keyword1, keyword2):  # 替换字典中key1为key2，值不变
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword1 in key:
            new_key = key.replace(keyword1, keyword2)
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def initialize_weights(modules, init_mode):         # 初始化权重
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',       # 根据输入的initMode 确定初始化权重的方法
                                        nonlinearity='relu')            # 恺明初始化
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))  # 返回错误信息
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)                                # 初始化b
        elif isinstance(m, nn.BatchNorm2d):                                 # 批归一化
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
