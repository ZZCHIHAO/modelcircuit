"""Serializable settings for a standalone segmentation prediction job."""
import re

PROFILE = 'segmentation-infer-v1'
REPORT_NAME = 'inference.json'
PENDING_REPORT = 'inference.pending.json'
_DEFAULTS = {'profile': PROFILE, 'run_spec': '', 'checkpoint': None, 'source': '', 'mode': 'fixed',
             'overlap': .25, 'device': 'cpu', 'max_runtime_seconds': 3600}


def validate_inference_config(config):
    if not isinstance(config, dict) or not config.keys() <= _DEFAULTS.keys():
        raise ValueError('Invalid segmentation inference settings')
    settings = {**_DEFAULTS, **config}
    if settings['profile'] != PROFILE:
        raise ValueError('Invalid segmentation inference profile')
    for name in ('run_spec', 'checkpoint', 'source'):
        value = settings[name]
        if name == 'checkpoint' and value is None:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > 4096 or '\0' in value:
            raise ValueError(f'Select a valid {name} path')
    if settings['mode'] not in ('fixed', 'sliding'):
        raise ValueError('Choose fixed or sliding segmentation inference')
    overlap = settings['overlap']
    if type(overlap) not in (float, int) or not 0 <= overlap <= .75:
        raise ValueError('Sliding window overlap must be in 0..0.75')
    if not isinstance(settings['device'], str) or not re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)', settings['device']):
        raise ValueError('Select CPU or one explicit CUDA device')
    seconds = settings['max_runtime_seconds']
    if type(seconds) is not int or not 10 <= seconds <= 604800:
        raise ValueError('Runtime limit must be 10..604800 seconds')
    return settings
