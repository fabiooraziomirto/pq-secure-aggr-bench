"""Shared simulator run logic for scripts; opt-in, never imported by tests of crypto."""
import importlib.metadata as md
import json
import resource

import numpy as np

from pqfl.aggregators import AGGREGATOR_REGISTRY
from pqfl.aggregators.insecure_stub import InsecureSumStub
from pqfl.aggregators.plain_fl import PlainFixedPoint, PlainFL
from pqfl.config import RunConfig
from pqfl.logging.metrics import JSONLWriter
from pqfl.net.replay import ledger_stages


def build_authenticator(cfg: RunConfig, sites):
    if cfg.auth_method == "none_insecure":
        from pqfl.auth import AuthProof

        class NoAuth:  # INSECURE: excludes authentication cost; label such runs.
            def issue(self, site_id, epoch, public_key):
                return AuthProof()

            def verify(self, *args):
                pass

        return NoAuth()
    from pqfl.auth import MLDSAAuth
    from pqfl.kem.backends import load_oqs
    oqs = load_oqs()
    public, secret = {}, {}
    for site in sites:
        with oqs.Signature("ML-DSA-65") as signer:
            public[site] = bytes(signer.generate_keypair())
            secret[site] = bytes(signer.export_secret_key())
    return MLDSAAuth(public, secret, {site: dict(public) for site in sites})


KEM_SCHEMES = ("A", "C", "A-dropout", "B-BGV-threshold")


def build_kem(cfg: RunConfig):
    if cfg.kem == "x25519_classical":
        from pqfl.kem.classical import ClassicalX25519KEM
        return ClassicalX25519KEM()
    from pqfl.kem.backends import LibOQSKEM
    from pqfl.kem.hybrid import HybridKEM
    return HybridKEM(LibOQSKEM())


def build_aggregator(scheme, cfg: RunConfig, sites, writer):
    if cfg.kem != "hybrid" and scheme not in KEM_SCHEMES:
        raise ValueError(f"kem={cfg.kem} is meaningless for {scheme}, which uses no KEM")
    if scheme == "STUB":
        return InsecureSumStub(writer=writer)
    if scheme == "N":
        return PlainFL(writer=writer)
    if scheme == "N-FP":
        return PlainFixedPoint(scale_bits=cfg.scale_bits, modulus_bits=cfg.modulus_bits, writer=writer)
    auth = build_authenticator(cfg, sites)
    if scheme == "B-BGV":
        return AGGREGATOR_REGISTRY["B-BGV"](clip_bound=cfg.clip_bound, scale_bits=cfg.bgv_scale_bits,
                                            ring_dim=cfg.ring_dim, plaintext_bits=cfg.bgv_plaintext_bits,
                                            authenticator=auth, writer=writer)
    if scheme == "B-BGV-threshold":
        threshold = cfg.dropout_threshold or max(2, len(sites) - 1)   # tolerates one dropout by default
        return AGGREGATOR_REGISTRY[scheme](build_kem(cfg), clip_bound=cfg.clip_bound, threshold=threshold,
                                           scale_bits=cfg.bgv_scale_bits, ring_dim=cfg.ring_dim,
                                           plaintext_bits=cfg.bgv_plaintext_bits, authenticator=auth, writer=writer)
    if scheme == "B":
        return AGGREGATOR_REGISTRY["B"](clip_bound=cfg.clip_bound, ring_dim=cfg.ring_dim,
                                        scaling_mod_size=cfg.scaling_mod_size,
                                        authenticator=auth, writer=writer)
    return AGGREGATOR_REGISTRY[scheme](build_kem(cfg), clip_bound=cfg.clip_bound,
                                       scale_bits=cfg.scale_bits, modulus_bits=cfg.modulus_bits,
                                       authenticator=auth, writer=writer)


def site_ids(count: int) -> list[str]:
    if type(count) is not int or count < 2:
        raise ValueError("Require at least two sites")
    return [f"S{index:04d}" for index in range(count)]


def run_cell(scheme, cfg: RunConfig, *, sites, vector_size, rounds, warmup, output,
             tolerance=1e-3, ledger_output=None, rekey_every=None, peak_memory_output=None):
    """One fresh session: setup (logged), `warmup` unlogged rounds, `rounds` logged rounds.

    Warm-up rounds consume the first round indices but write no rows, so logged
    rounds start at index `warmup`. Updates come from the public run_seed.
    """
    if rounds <= 0 or warmup < 0 or vector_size <= 0:
        raise ValueError("rounds and vector_size must be positive, warmup nonnegative")
    versions = {}
    for name in ("liboqs-python", "openfhe"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    writer = JSONLWriter(output, run_id=f"{cfg.run_id}-{scheme}-n{len(sites)}-v{vector_size}",
                         config_hash=cfg.config_hash(scheme), scheme=scheme, dp_level=cfg.dp_level,
                         run_seed=cfg.run_seed, library_versions=versions)
    aggregator = build_aggregator(scheme, cfg, sites, writer)
    rekey_every = rekey_every if rekey_every is not None else cfg.rekey_every_rounds
    rng = np.random.default_rng(cfg.run_seed)
    aggregator.setup(list(sites))
    setup_stages = ledger_stages(aggregator.stats())
    round_stages = None
    for round_idx in range(warmup + rounds):
        # Metrics read aggregator.writer when each operation starts; None silences warm-up.
        aggregator.writer = writer if round_idx >= warmup else None
        if rekey_every and round_idx and round_idx % rekey_every == 0:
            aggregator.rekey()
        updates = {site: rng.uniform(-cfg.clip_bound, cfg.clip_bound, vector_size) for site in sites}
        result = aggregator.aggregate_round(round_idx, updates)
        round_stages = ledger_stages(aggregator.stats())
        error = float(np.max(np.abs(result - sum(updates.values()))))
        if error > tolerance:
            raise RuntimeError(f"Round {round_idx}: aggregate error {error:.3e} exceeds {tolerance}")
    if peak_memory_output is not None:
        # Whole simulator process (all actors and the coordinator), not one actor.
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(peak_memory_output, "x", encoding="utf-8") as handle:
            json.dump(dict(peak_host_rss_mb=peak_kb / 1024, scope="whole simulator process"), handle)
    if ledger_output is not None:
        # Sizes only: the ledger holds no payload. Round sizes are constant across rounds.
        with open(ledger_output, "x", encoding="utf-8") as handle:
            json.dump(dict(scheme=scheme, sites=list(sites), vector_size=vector_size,
                           setup=setup_stages, round=round_stages), handle)
