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
if command -v tc >/dev/null 2>&1; then
    tc -V
else
    echo 'tc missing; privileges for network emulation also require separate verification.'
    failed=1
fi
exit "$failed"
