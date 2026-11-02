import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("tls_baseline", Path("scripts/tls_baseline.py"))
tls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tls)

TRACE = """>>> TLS 1.3, Handshake [length 04c6], ClientHello
 16 03 01 04 c6
<<< TLS 1.3, Handshake [length 0425], ServerHello
<<< TLS 1.3, ChangeCipherSpec [length 0001], ChangeCipherSpec
<<< TLS 1.3, Handshake [length 0020], EncryptedExtensions
<<< TLS 1.3, Handshake [length 0a20], Certificate
>>> TLS 1.3, Handshake [length 0035], Finished
"""


def test_handshake_bytes_are_summed_per_direction_and_only_for_handshake_records():
    parsed = tls.parse_handshake_bytes(TRACE)
    assert parsed["client_to_server"] == 0x4c6 + 0x35
    assert parsed["server_to_client"] == 0x425 + 0x20 + 0xa20
    assert parsed["total"] == parsed["client_to_server"] + parsed["server_to_client"] and parsed["records"] == 6
    assert tls.parse_handshake_bytes("nothing")["total"] == 0
