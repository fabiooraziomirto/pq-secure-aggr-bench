"""Public runtime metadata only; never inspect environment variable values."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

from pqfl.kem.backends import BackendUnavailable, load_oqs


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_metadata() -> dict:
    cpu = platform.processor() or None
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    result = dict(python=platform.python_version(), cryptography=_version('cryptography'),
                  cpu_model=cpu, liboqs=None, liboqs_python=_version('liboqs-python'),
                  enabled_kems=[], enabled_sigs=[], oqs_available=False)
    try:
        oqs = load_oqs()
    except BackendUnavailable:
        return result
    result.update(liboqs=oqs.oqs_version(), liboqs_python=oqs.oqs_python_version(),
                  enabled_kems=list(oqs.get_enabled_kem_mechanisms()),
                  enabled_sigs=list(oqs.get_enabled_sig_mechanisms()), oqs_available=True)
    return result
