"""Bounded, explicit refresh of local JSON architecture chips.

The catalog holds a validated content snapshot. Drag payloads contain its
content ID, never a filesystem path or an executable plugin name.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from studio_core.component_types import ComponentBundle
from .chip_library import MAX_BYTES, read_bundle


@dataclass
class LocalComponentSpec:
    id: str
    label: str
    description: str
    config_defaults: dict
    input_ports: tuple
    output_ports: tuple
    source_names: tuple


class LocalComponentIndex:
    def __init__(self, directory, *, max_files=64, max_total_bytes=16 * 1024 * 1024):
        if type(max_files) is not int or not 1 <= max_files <= 64:
            raise ValueError('Local library file limit must be an integer in 1..64')
        if type(max_total_bytes) is not int or not 1 <= max_total_bytes <= 16 * 1024 * 1024:
            raise ValueError('Local library byte limit must be an integer in 1..16777216')
        self.directory = Path(directory)
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self.diagnostics = []
        self._entries = {}
        self.initialized = False

    def refresh(self):
        entries, diagnostics, candidates = {}, [], []
        try:
            with os.scandir(self.directory) as iterator:
                for seen, item in enumerate(iterator):
                    if seen >= 512:
                        diagnostics.append('目录超过512项，已停止扫描；请将芯片放入独立库目录。')
                        break
                    if not item.name.casefold().endswith('.json'):
                        continue
                    if item.is_symlink():
                        diagnostics.append(item.name + '：不读取链接文件，请复制实际芯片包。')
                    elif item.is_file(follow_symlinks=False):
                        candidates.append(Path(item.path))
        except FileNotFoundError:
            pass
        except OSError as error:
            diagnostics.append('无法读取本地芯片目录：' + str(error))
        candidates.sort(key=lambda path: (path.name.casefold(), path.name))
        if len(candidates) > self.max_files:
            diagnostics.append('本地芯片文件超过' + str(self.max_files) + '项，仅读取排序靠前的文件。')
        total = 0
        for path in candidates[:self.max_files]:
            try:
                size = path.stat().st_size
                if size > MAX_BYTES:
                    diagnostics.append(path.name + '：Chip bundle exceeds 4 MiB，已跳过未读取的文件。')
                    continue
                if total + size > self.max_total_bytes:
                    diagnostics.append(path.name + '：超过本次读取预算，请缩小芯片库或移出较大文件。')
                    continue
                total += size
                payload = read_bundle(path)
                encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode('utf-8')
                identifier = 'local:' + hashlib.sha256(encoded).hexdigest()
                if identifier in entries:
                    entries[identifier]['sources'].append(path.name)
                    continue
                entries[identifier] = {'payload': payload, 'sources': [path.name]}
            except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
                diagnostics.append(path.name + '：' + str(error))
        self._entries = entries
        self.diagnostics = diagnostics
        self.initialized = True
        return self.list_components()

    def list_components(self, query=''):
        if type(query) is not str:
            raise ValueError('Local component query must be a string')
        words = query.casefold().split()
        result = []
        for identifier, entry in self._entries.items():
            payload = entry['payload']
            definition = next(item for item in payload['definitions'] if item['id'] == payload['definition_id'])
            name = definition.get('name')
            label = (name if type(name) is str and name else definition['id']) + ' v' + definition['version']
            source_names = tuple(entry['sources'])
            haystack = ' '.join((identifier, label, definition['id'], *source_names)).casefold()
            if not all(word in haystack for word in words):
                continue
            result.append(LocalComponentSpec(identifier, label,
                '本地架构芯片 · 可展开/前向/训练/Python往返；只包含架构，无权重。',
                {item['id']: deepcopy(item['default']) for item in definition.get('public_parameters', [])},
                tuple(port['id'] for port in definition['graph']['inputs']),
                tuple(port['id'] for port in definition['graph']['outputs']), source_names))
        return result

    def build(self, identifier, config):
        from .document import _json_values
        from .component_insert import _check_entry
        if type(identifier) is not str or identifier not in self._entries:
            raise ValueError('本地芯片快照已变化，请刷新并重新选择。')
        if type(config) is not dict:
            raise ValueError('Local component configuration must be an object')
        _json_values(config)
        payload = self._entries[identifier]['payload']
        requirements = payload.get('requirements', {'operator_set': 'vision-ops-v4',
                                                     'resource_profile': 'classification-v2'})
        definitions = deepcopy(payload['definitions'])
        definition = next(item for item in definitions if item['id'] == payload['definition_id'])
        entry = {'kind': 'chip', 'definition_id': payload['definition_id'], 'params': deepcopy(config)}
        _check_entry(definition, entry, definitions, requirements)
        return ComponentBundle(entry, definitions, deepcopy(requirements))
