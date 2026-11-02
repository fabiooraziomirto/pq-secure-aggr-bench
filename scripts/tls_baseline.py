#!/usr/bin/env python3
"""Baseline: TLS 1.3 handshakes with the local OpenSSL (opt-in; never run by tests).

Measures, over loopback, the wall time of `openssl s_client` handshakes and the handshake message bytes
(from `-msg`) for classical X25519, the standard hybrid X25519MLKEM768 and, for the certificate, ECDSA P-256
versus ML-DSA-65. Process start-up cost is measured with `openssl version` and subtracted. Not a network
measurement and not comparable with the application-level HybridKEM except for the key-share sizes.
"""
import argparse
import json
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

CONFIG = "[req]\ndistinguished_name = dn\nprompt = no\nx509_extensions = v3\n[dn]\nCN = localhost\n" \
         "[v3]\nbasicConstraints = critical,CA:TRUE\n"
MESSAGE = re.compile(r"([<>]{3}) TLS 1\.[23], (\w+) \[length ([0-9a-f]{4})\]")


def parse_handshake_bytes(text: str) -> dict:
    """Sum of the [length] of Handshake messages per direction, and the message names, from `s_client -msg`."""
    sizes, names = {"client_to_server": 0, "server_to_client": 0}, []
    for arrow, kind, length in MESSAGE.findall(text):
        if kind == "Handshake":
            sizes["client_to_server" if arrow == ">>>" else "server_to_client"] += int(length, 16)
        names.append((arrow, kind))
    return dict(sizes, total=sum(sizes.values()), records=len(names))


def make_certificate(openssl: str, directory: Path, algorithm: str):
    config = directory / "openssl.cnf"
    config.write_text(CONFIG)
    key, cert = directory / f"{algorithm}.key", directory / f"{algorithm}.pem"
    newkey = "ec" if algorithm == "ecdsa" else "ML-DSA-65"          # "mldsa" means ML-DSA-65
    extra = ["-pkeyopt", "ec_paramgen_curve:P-256"] if algorithm == "ecdsa" else []
    subprocess.run([openssl, "req", "-x509", "-config", str(config), "-newkey", newkey, *extra, "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-days", "1"], check=True, capture_output=True)
    return cert, key


def run_case(openssl, directory, port, group, algorithm, repetitions):
    cert, key = make_certificate(openssl, directory, algorithm)
    server = subprocess.Popen([openssl, "s_server", "-accept", str(port), "-cert", str(cert), "-key", str(key),
                               "-groups", group, "-tls1_3", "-quiet", "-no_ticket"], stdin=subprocess.PIPE,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    client = [openssl, "s_client", "-connect", f"localhost:{port}", "-groups", group, "-tls1_3", "-brief"]
    try:
        traced = subprocess.run(client + ["-msg"], input=b"", capture_output=True, timeout=30)
        sizes = parse_handshake_bytes(traced.stdout.decode(errors="replace") + traced.stderr.decode(errors="replace"))
        times = []
        for _ in range(repetitions):
            start = time.perf_counter()
            subprocess.run(client, input=b"", capture_output=True, timeout=30)
            times.append(time.perf_counter() - start)
    finally:
        server.kill()
    return sizes, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openssl", required=True, help="Path of the OpenSSL >= 3.5 executable")
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--output", required=True, help="New JSON file; never overwritten")
    args = parser.parse_args()
    baseline = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        subprocess.run([args.openssl, "version"], capture_output=True)
        baseline.append(time.perf_counter() - start)
    cases = [("x25519", "ecdsa"), ("X25519MLKEM768", "ecdsa"), ("X25519MLKEM768", "ML-DSA-65"), ("MLKEM768", "ML-DSA-65")]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, (group, algorithm) in enumerate(cases):
            cert_name = "ecdsa" if algorithm == "ecdsa" else "mldsa"
            directory = Path(tmp)
            sizes, times = run_case(args.openssl, directory, args.base_port + index, group,
                                    "ecdsa" if algorithm == "ecdsa" else "mldsa", args.repetitions)
            handshake = statistics.median(times) - statistics.median(baseline)
            results.append(dict(group=group, certificate=algorithm, handshake_bytes=sizes,
                                median_connection_s=statistics.median(times), process_start_s=statistics.median(baseline),
                                median_handshake_s=handshake, samples=len(times)))
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(dict(openssl=subprocess.run([args.openssl, "version"], capture_output=True, text=True).stdout.strip(),
                       results=results), handle, indent=1)
    for r in results:
        print(f"{r['group']:16s} {r['certificate']:10s} bytes {r['handshake_bytes']['total']:>6d} "
              f"(c->s {r['handshake_bytes']['client_to_server']}, s->c {r['handshake_bytes']['server_to_client']}) "
              f"handshake {1000 * r['median_handshake_s']:.1f} ms")


if __name__ == "__main__":
    main()
