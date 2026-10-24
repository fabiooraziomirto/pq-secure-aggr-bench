#!/usr/bin/env bash
# Diagnostic only: execute on the target cluster. No setup or benchmark.
set -u
failed=0
if command -v openssl >/dev/null 2>&1; then
    openssl version
    if algorithms=$(openssl list -kem-algorithms 2>&1); then
        if printf '%s\n' "$algorithms" | grep -Ei 'ML-KEM|MLKEM'; then
            echo 'ML-KEM listed; binding and ML-KEM-768 interoperability still need validation.'
        else
            echo 'ML-KEM not listed by active OpenSSL providers.'
            failed=1
        fi
    else
        printf '%s\n' "$algorithms"
        failed=1
    fi
else
    echo 'OpenSSL missing.'
    failed=1
fi
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists liboqs; then
    pkg-config --modversion liboqs
else
    echo 'liboqs not discoverable via pkg-config; inspect cluster modules/library paths.'
    failed=1
fi
# Use the same lazy prerequisite checks as the application; do not download bindings.
if ! python - <<'PYTHON'
from pqfl.kem.backends import BackendUnavailable, load_oqs
try:
    oqs = load_oqs()
except BackendUnavailable as exc:
    print(str(exc))
    raise SystemExit(1)
print("liboqs:", oqs.oqs_version())
print("liboqs-python:", oqs.oqs_python_version())
mechanisms = [name for name in oqs.get_enabled_kem_mechanisms()
              if "ML-KEM" in name.upper() or "KYBER" in name.upper()]
print("Enabled ML-KEM / legacy Kyber mechanisms:", mechanisms)
if "ML-KEM-768" not in mechanisms:
    print("Standardized ML-KEM-768 is not enabled; Kyber768 is a distinct legacy mechanism.")
    raise SystemExit(1)
PYTHON
then
    failed=1
fi
if command -v tc >/dev/null 2>&1; then
    tc -V
else
    echo 'tc missing; privileges for network emulation also require separate verification.'
    failed=1
fi
exit "$failed"
