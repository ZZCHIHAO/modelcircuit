"""Bounded, explicitly requested observation of real tensor values."""
import re
from typing import TypedDict

MAX_BYTES = 1_048_576
MAX_SOURCES = 16
MAX_SIDE = 64
_REFERENCE = re.compile(r'(?:call:[A-Za-z_]\w*/)*(?:input:[A-Za-z_]\w*|call:[A-Za-z_]\w*(?::[A-Za-z_]\w*)?)', re.ASCII)


class ProbeRequest(TypedDict):
    source: str
    sample_index: int
    head_index: int
    query_range: list[int]
    key_range: list[int]
    statistics: list[str]


class CapturePolicy(TypedDict):
    requests: list[ProbeRequest]
    max_sources: int
    max_bytes: int
    max_attention_side: int


def normalize_probe_requests(values):
    if not isinstance(values, list) or len(values) > MAX_SOURCES:
        raise ValueError('attention_probe_requests must be a list of at most 16 requests')
    result, seen = [], set()
    for item in values:
        if not isinstance(item, dict) or set(item) != set(ProbeRequest.__annotations__):
            raise ValueError('Invalid attention probe request fields')
        source = item['source']
        if not isinstance(source, str) or len(source) > 1024 or not _REFERENCE.fullmatch(source) or source in seen:
            raise ValueError('Probe source must be a distinct tensor reference')
        seen.add(source)
        for key in ('sample_index', 'head_index'):
            if type(item[key]) is not int or not 0 <= item[key] <= 2**31 - 1:
                raise ValueError(f'{source}: invalid {key}')
        for key in ('query_range', 'key_range'):
            span = item[key]
            if (not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span)
                    or not 0 <= span[0] < span[1] <= 2**31 - 1):
                raise ValueError(f'{source}: invalid {key}')
        stats = item['statistics']
        if (not isinstance(stats, list) or len(stats) > 4 or any(s not in ('mean', 'min', 'max', 'norm') for s in stats)
                or len(set(stats)) != len(stats)):
            raise ValueError(f'{source}: invalid probe statistics')
        result.append({**item, 'query_range': list(item['query_range']), 'key_range': list(item['key_range']), 'statistics': list(stats)})
    return result


def capture_tensor(value, spec, request, *, remaining_bytes, max_side=MAX_SIDE):
    """Index first; only the bounded rectangle can reach the host."""
    import torch

    source = request['source']
    sample, head = request['sample_index'], request['head_index']
    q, k = request['query_range'], request['key_range']
    base = {'shape': list(value.shape), 'dtype': spec['dtype'], 'axes': list(spec['axes']), 'source': 'measured',
            'count': value.numel(), 'values': [], 'statistics': {}, 'captured_bytes': 0, 'omitted': True, 'truncated': False}
    if value.ndim != 4 or spec['axes'][1:] != ['head', 'query', 'key']:
        return dict(base, reason='Expected [batch, head, query, key] attention tensor')
    if not (sample < value.shape[0] and head < value.shape[1] and q[0] < value.shape[2] and k[0] < value.shape[3]):
        return dict(base, reason=f'{source}: sample, head or slice starts outside measured tensor')
    q_end, k_end = min(q[1], q[0] + max_side, value.shape[2]), min(k[1], k[0] + max_side, value.shape[3])
    count = (q_end - q[0]) * (k_end - k[0])
    nbytes = count * value.element_size()
    if nbytes > remaining_bytes:
        return dict(base, reason='Capture byte budget exhausted')
    with torch.no_grad():
        snapshot = value[sample, head, q[0]:q_end, k[0]:k_end].detach().to(device='cpu', copy=True)
        if not bool(torch.isfinite(snapshot).all()):
            return dict(base, reason='Selected slice contains non-finite values')
        reduction = snapshot.double()
        stats = {key: float(getattr(reduction, key)()) for key in request['statistics']}
        return dict(base, selection={'sample_index': sample, 'head_index': head, 'query_range': [q[0], q_end],
                                    'key_range': [k[0], k_end]}, values=snapshot.tolist(), statistics=stats,
                    min=stats.get('min'), max=stats.get('max'), mean=stats.get('mean'), std=None,
                    captured_bytes=nbytes, sample_size=count, omitted=False, truncated=q[1] != q_end or k[1] != k_end)


class _Capture:
    def __init__(self, requests, max_bytes, max_side):
        self.requests = {request['source']: request for request in requests}
        self.remaining = max_bytes
        self.max_side = max_side
        self.results = {source: {'omitted': True, 'reason': 'Tensor was not observed', 'values': [], 'captured_bytes': 0}
                        for source in self.requests}

    def observe(self, source, value, spec):
        if source not in self.requests:
            return
        result = capture_tensor(value, spec, self.requests[source], remaining_bytes=self.remaining, max_side=self.max_side)
        self.remaining -= result['captured_bytes']
        self.results[source] = result


def make_capture(policy, resolved):
    if not isinstance(policy, dict) or not set(policy) <= set(CapturePolicy.__annotations__):
        raise ValueError('Invalid capture policy fields')
    settings = [policy.get('max_sources', MAX_SOURCES), policy.get('max_bytes', MAX_BYTES),
                policy.get('max_attention_side', MAX_SIDE)]
    if any(type(n) is not int or not 1 <= n <= ceiling for n, ceiling in zip(settings, (MAX_SOURCES, MAX_BYTES, MAX_SIDE))):
        raise ValueError('Capture policy exceeds source, byte or attention slice limits')
    requests = normalize_probe_requests(policy.get('requests', []))
    if len(requests) > settings[0]:
        raise ValueError('Capture policy exceeds its source limit')
    return _Capture(requests, settings[1], settings[2])
