"""Serializable requests for the desktop's bounded diagnostic runs."""
from copy import deepcopy
import math
import re

PROFILE = 'visual-debug-v1'
MAX_PROBES = 12
MAX_SWEEP = 8
DATA_KINDS = frozenset({'csv', 'regression_csv', 'image_folder'})
IDENTIFIER = r'[A-Za-z_][A-Za-z0-9_]*'
SOURCE = re.compile(rf'(?:call:{IDENTIFIER}/){{0,8}}(?:input:{IDENTIFIER}|call:{IDENTIFIER}(?::{IDENTIFIER})?)')


def numeric_parameters(graph, instance):
    params = {}
    if instance['kind'] == 'chip':
        definition = next(item for item in graph.get('definitions', []) if item['id'] == instance['definition_id'])
        params.update({item['id']: item.get('default') for item in definition.get('public_parameters', [])})
    params.update(instance.get('params', {}))
    return {name: value for name, value in params.items() if type(value) in (int, float)}


def validate_debug_config(config):
    if not isinstance(config, dict):
        raise ValueError('调试配置必须为对象')
    result = {'profile': PROFILE, 'mode': 'forward', 'device': 'cpu', 'batch_size': 2,
              'seed': 42, 'lr': 0.01, 'optimizer': 'sgd', 'sources': [], 'probe_offset': 0, 'experiment': None,
              'split': 'train', 'sample_ids': [], 'sweep': None}
    if not config.keys() <= result.keys():
        raise ValueError('未知调试选项')
    result.update(deepcopy(config))
    if result['profile'] != PROFILE or result['mode'] not in ('forward', 'step', 'sweep'):
        raise ValueError('未知调试模式')
    device = result['device']
    if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda:(?:0|[1-9][0-9]*)', device):
        raise ValueError('请选择 CPU 或一张 CUDA 显卡')
    for field, lower, upper in [('seed', 0, 2**31 - 1), ('batch_size', 1, 64), ('probe_offset', 0, 2**31 - 1)]:
        if type(result[field]) is not int or not lower <= result[field] <= upper:
            raise ValueError(f'{field} 须为 {lower} 到 {upper} 的整数')
    if type(result['lr']) not in (int, float) or not 1e-6 <= result['lr'] <= 1:
        raise ValueError('单步学习率须在 0.000001 至 1 之间')
    if result['optimizer'] not in ('sgd', 'adamw'):
        raise ValueError('请选择 SGD 或 AdamW')
    sources = result['sources']
    if (not isinstance(sources, list) or len(sources) > MAX_PROBES or
            any(not isinstance(ref, str) or len(ref) > 1024 or not SOURCE.fullmatch(ref) for ref in sources)):
        raise ValueError(f'请选择最多 {MAX_PROBES} 个有效端口作为探针')
    result['sources'] = list(dict.fromkeys(sources))
    split, identifiers = result['split'], result['sample_ids']
    if split not in ('train', 'val', 'test'):
        raise ValueError('请选择训练、验证或测试划分')
    if (not isinstance(identifiers, list) or len(identifiers) > result['batch_size']
            or not all(isinstance(identifier, str) and identifier and len(identifier) <= 4096 for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('样本 ID 必须唯一，且数量不能超过调试样本数')
    if result['experiment'] is not None:
        from .experiment_spec import validate_supervised_config
        experiment = validate_supervised_config(result['experiment'])
        if (experiment['profile'] not in ('classification-v2', 'regression-v1')
                or experiment['data']['kind'] not in DATA_KINDS):
            raise ValueError('真实样本调试支持数值 CSV 分类、数值 CSV 回归和 ImageFolder 图片分类')
        if result['mode'] == 'step' and split != 'train':
            raise ValueError('训练一步请选择训练集。验证集和测试集可用于样例前向。')
        result['experiment'] = experiment
    elif identifiers or split != 'train':
        raise ValueError('请先选择已配置的实验数据')
    sweep = result['sweep']
    if result['mode'] == 'sweep':
        if not isinstance(sweep, dict) or set(sweep) != {'instance', 'parameter', 'values'}:
            raise ValueError('请选择待扫描的组件实例和一个数值参数')
        if any(not isinstance(sweep[key], str) or not re.fullmatch(IDENTIFIER, sweep[key])
               for key in ('instance', 'parameter')):
            raise ValueError('参数扫描需要有效的实例 ID 和参数名')
        values = sweep['values']
        if (not isinstance(values, list) or not 2 <= len(values) <= MAX_SWEEP or
                any(type(value) not in (int, float) or not -1e12 <= value <= 1e12 or not math.isfinite(value)
                    for value in values) or len(set(values)) != len(values)):
            raise ValueError(f'请输入 2 至 {MAX_SWEEP} 个不同的有限数值')
    elif sweep is not None:
        raise ValueError('参数扫描设置只能用于扫描模式')
    return result


def compare_debug_results(left, right):
    """Compare measurements only when both runs used the same inputs and settings."""
    if left.get('mode') not in ('forward', 'step') or left.get('mode') != right.get('mode'):
        return {'comparable': False, 'reason': '请选择同一调试模式的两次测量'}
    for fields, reason in (
        (('batch_digest', 'dataset_digest', 'batch_kind', 'split', 'input_shapes', 'input_dtypes', 'task_profile'),
         '两次测量使用的样本、划分或输入不同'),
        (('seed', 'device', 'optimizer_settings', 'probe_offset'), '两次测量的种子、设备、优化器或采样范围不同'),
    ):
        if any(left.get(key) != right.get(key) for key in fields):
            return {'comparable': False, 'reason': reason}
    if not left.get('batch_digest'):
        return {'comparable': False, 'reason': '测量缺少样本记录'}
    a, b = left['loss'], right['loss']
    return {'comparable': True, 'same_model': left['model_digest'] == right['model_digest'],
            'loss_delta': b - a if a is not None and b is not None else None}
