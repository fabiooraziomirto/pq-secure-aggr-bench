"""Cross-implementation functional checks, gated on observed CLI capabilities."""
import re
import shutil
import subprocess

import pytest

from pqfl.kem.backends import BackendUnavailable, LibOQSKEM


def _run(executable, *args):
    result = subprocess.run([executable, *map(str, args)], capture_output=True, timeout=30)
    if result.returncode:
        # Never include key material or subprocess buffers in failure output.
        pytest.fail("OpenSSL interop subcommand failed; inspect the cluster installation")
    return result.stdout + result.stderr


def _cli_capabilities():
    executable = shutil.which("openssl")
    if executable is None:
        pytest.skip("TODO cluster interop: OpenSSL executable missing")
    help_text = _run(executable, "pkeyutl", "-help").decode(errors="replace")
    # These options were observed in pkeyutl -help before writing this test.
    required = ("encap", "decap", "secret", "pubin", "inkey", "in", "out")
    if any(re.search(r"(?m)^\s*-" + flag + r"\s", help_text) is None for flag in required):
        pytest.skip("TODO cluster interop: pkeyutl -help lacks the required encaps/decaps options")
    mechanisms = _run(executable, "list", "-kem-algorithms").decode(errors="replace")
    if re.search(r"\bML-KEM-768\b", mechanisms) is None:
        pytest.skip("TODO cluster interop: OpenSSL does not advertise ML-KEM-768")
    return executable


def _tlv(data, offset, tag):
    """Read DER framing only; no cryptographic algorithm or hand-coded OID."""
    if offset + 2 > len(data) or data[offset] != tag:
        raise ValueError("Unexpected OpenSSL SPKI framing")
    size = data[offset + 1]
    start = offset + 2
    if size & 128:
        width = size & 127
        if not 1 <= width <= 4 or start + width > len(data):
            raise ValueError("Invalid DER length")
        size = int.from_bytes(data[start:start + width], "big")
        start += width
    end = start + size
    if end > len(data):
        raise ValueError("Truncated DER")
    return start, end


def _public_bitstring(der):
    start, end = _tlv(der, 0, 0x30)
    if end != len(der):
        raise ValueError("Trailing SPKI data")
    _, algorithm_end = _tlv(der, start, 0x30)
    bits_start, bits_end = _tlv(der, algorithm_end, 0x03)
    if bits_end != end or bits_start == bits_end or der[bits_start] != 0:
        raise ValueError("Invalid SPKI bit string")
    raw_start = bits_start + 1
    if bits_end - raw_start != 1184:
        raise ValueError("OpenSSL public key is not the expected ML-KEM-768 size")
    return raw_start


def test_mlkem_liboqs_openssl_interoperability(tmp_path):
    executable = _cli_capabilities()
    try:
        kem = LibOQSKEM()
    except BackendUnavailable as exc:
        pytest.skip("TODO cluster interop: " + str(exc))
    directory = tmp_path / "interop"
    directory.mkdir(mode=0o700)
    private = directory / "openssl.private.pem"
    public_file = directory / "openssl.public.der"
    ciphertext_file = directory / "ciphertext.bin"
    shared_file = directory / "shared.bin"
    # These genpkey/pkey options were checked in their local -help output.
    _run(executable, "genpkey", "-algorithm", "ML-KEM-768", "-out", private)
    _run(executable, "pkey", "-in", private, "-pubout", "-outform", "DER", "-out", public_file)
    template = public_file.read_bytes()
    raw_start = _public_bitstring(template)
    ciphertext, shared = kem.encaps(template[raw_start:])
    ciphertext_file.write_bytes(ciphertext)
    _run(executable, "pkeyutl", "-decap", "-inkey", private,
         "-in", ciphertext_file, "-out", shared_file)
    if shared_file.read_bytes() != shared:
        pytest.fail("liboqs encaps / OpenSSL decaps mismatch; secrets suppressed")
    public, secret = kem.keygen()
    if len(public) != len(template) - raw_start:
        pytest.fail("Incompatible public key lengths")
    # Preserve the algorithm identifier and DER headers exported by OpenSSL for
    # this mechanism. Replace only the same-length public BIT STRING payload.
    public_file.write_bytes(template[:raw_start] + public)
    _run(executable, "pkeyutl", "-encap", "-pubin", "-inkey", public_file,
         "-out", ciphertext_file, "-secret", shared_file)
    recovered = kem.decaps(secret, ciphertext_file.read_bytes())
    if recovered != shared_file.read_bytes():
        pytest.fail("OpenSSL encaps / liboqs decaps mismatch; secrets suppressed")
