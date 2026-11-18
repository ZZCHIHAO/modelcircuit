"""Runtime capabilities are different from hardware discovery. No torch at import."""
import csv
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import re
import shutil
import subprocess


def _hardware_gpus():
    executable=shutil.which('nvidia-smi')
    if not executable:return []
    try:
        result=subprocess.run([executable,'--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader,nounits'],
                              capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=5,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:return []
        hardware=[]
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row)==4:
                hardware.append({'index':int(row[0].strip()),'name':row[1].strip(),
                                 'total_memory_mb':int(row[2].strip()),'driver_version':row[3].strip()})
        return hardware
    except (OSError,ValueError,subprocess.TimeoutExpired):return []


def device_capabilities():
    result={'queried_at':datetime.now(timezone.utc).isoformat(),'torch_version':None,'cuda_version':None,
            'cuda_available':False,'reason':None,'multi_gpu_supported':False,'devices':[],
            'hardware_gpus':_hardware_gpus()}
    try:
        import torch
        result['torch_version']=torch.__version__
        result['cuda_version']=torch.version.cuda
        result['devices'].append({'id':'cpu','name':'CPU','available':True})
        result['cuda_available']=bool(torch.cuda.is_available())
        if result['cuda_available']:
            for index in range(torch.cuda.device_count()):
                properties=torch.cuda.get_device_properties(index)
                result['devices'].append({'id':f'cuda:{index}','name':properties.name,'available':True,
                                          'total_memory_mb':int(properties.total_memory/(1024*1024))})
        elif torch.version.cuda is None:
            result['reason']='当前 PyTorch 为 CPU 构建；检测到显卡不代表此运行时支持 CUDA。'
        else:
            result['reason']='当前 CUDA 运行时不可用，请检查驱动、设备可见性与所选计算环境。'
    except Exception as error:
        result['reason']='无法查询 PyTorch 运行时：'+str(error)
    return result


def resolve_device(requested):
    if not isinstance(requested,str) or re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)',requested) is None:
        raise ValueError('device must be cpu or one explicit cuda:N; multi-GPU is not implemented')
    import torch
    if requested=='cpu':return torch.device('cpu')
    if not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable in the current PyTorch runtime; CPU fallback is disabled')
    index=int(requested.split(':')[1])
    if index>=torch.cuda.device_count():raise ValueError('Requested CUDA ordinal is not available: '+requested)
    return torch.device('cuda',index)


def _host_memory_snapshot():
    """Available physical memory bounded by process/container commit headroom."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD),
                        ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                        ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                        ('avail_extended_virtual', ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('GlobalMemoryStatusEx failed')
        return {'available_bytes': min(status.avail_phys, status.avail_page),
                'total_bytes': status.total_phys, 'physical_available_bytes': status.avail_phys,
                'commit_available_bytes': status.avail_page, 'source': 'GlobalMemoryStatusEx physical+commit'}
    if os.name == 'posix' and Path('/proc/meminfo').is_file():
        entries = {}
        for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
            key, value = line.split(':', 1)
            entries[key] = int(value.strip().split()[0]) * 1024
        available, total = entries['MemAvailable'], entries['MemTotal']
        cgroup_headrooms = []
        # Resolve this process's hierarchy, including nested cgroup v1/v2.
        paths = [(Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'),
                 (Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes')]
        cgroup = Path('/proc/self/cgroup')
        if cgroup.is_file():
            for line in cgroup.read_text(encoding='ascii').splitlines():
                _, controllers, relative = line.split(':', 2)
                if not controllers:
                    root = Path('/sys/fs/cgroup')
                    location = root / relative.lstrip('/')
                    paths.extend((parent, 'memory.max', 'memory.current') for parent in (location, *location.parents) if parent == root or root in parent.parents)
                elif 'memory' in controllers.split(','):
                    root = Path('/sys/fs/cgroup/memory')
                    location = root / relative.lstrip('/')
                    paths.extend((parent, 'memory.limit_in_bytes', 'memory.usage_in_bytes') for parent in (location, *location.parents) if parent == root or root in parent.parents)
        for root, limit_name, usage_name in paths:
            limit_path, usage_path = root / limit_name, root / usage_name
            if limit_path.is_file() and usage_path.is_file():
                limit = limit_path.read_text(encoding='ascii').strip()
                if limit != 'max':
                    cgroup_headrooms.append(max(0, int(limit) - int(usage_path.read_text(encoding='ascii').strip())))
        if cgroup_headrooms: available = min(available, *cgroup_headrooms)
        return {'available_bytes': available, 'total_bytes': total,
                'physical_available_bytes': entries['MemAvailable'],
                'cgroup_headroom_bytes': min(cgroup_headrooms) if cgroup_headrooms else None,
                'source': 'Linux MemAvailable+cgroup headroom'}
    raise OSError('Host available memory query is unsupported on this platform')


def memory_snapshot(requested):
    """Runtime-only free bytes for the selected device, with no CPU fallback."""
    device = resolve_device(str(requested))
    host = _host_memory_snapshot()
    result = {'device': str(device), 'queried_at': datetime.now(timezone.utc).isoformat(),
              'host_available_bytes': int(host['available_bytes']), 'host_total_bytes': int(host['total_bytes']),
              'host': host}
    if device.type == 'cpu':
        result.update(available_bytes=int(host['available_bytes']), total_bytes=int(host['total_bytes']), source=host['source'])
    else:
        import torch
        free, total = torch.cuda.mem_get_info(device)
        result.update(available_bytes=int(free), total_bytes=int(total), source='torch.cuda.mem_get_info(selected ordinal)')
    return result
