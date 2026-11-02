#!/usr/bin/env python3
"""Replay a ledger JSON over running agents and write per-repetition network times.

The ledger comes from `run_experiment.py --ledger-output`. Opt-in; every count is
explicit. Network time excludes compute and is not an end-to-end latency.
"""
import argparse
import asyncio
import json

from pqfl.net.replay import Controller


def parse_agent(text):
    actor, address = text.split("=", 1)
    host, port = address.rsplit(":", 1)
    return actor, (host, int(port))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--agents", nargs="+", required=True, metavar="ACTOR=HOST:PORT")
    parser.add_argument("--profile", required=True, help="Label of the emulated network profile")
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--output", required=True, help="New JSON file; never overwritten")
    parser.add_argument("--iface", default="eth0", help="Interface whose byte counters are recorded")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.warmup < 0:
        parser.error("repetitions must be positive and warmup nonnegative")
    return args


async def run(args):
    ledger = json.load(open(args.ledger, encoding="utf-8"))
    controller = Controller(dict(parse_agent(a) for a in args.agents))
    result = dict(profile=args.profile, scheme=ledger["scheme"], sites=ledger["sites"],
                  vector_size=ledger["vector_size"], repetitions=args.repetitions,
                  warmup=args.warmup)
    try:
        result["tcp"] = await controller.sysctl(next(iter(controller.addresses)))
        for part in ("setup", "round"):
            stages = [[tuple(t) for t in stage] for stage in ledger[part]]
            await controller.prepare(stages)
            wire_before = await controller.ifstats(args.iface)
            samples = []
            for repetition in range(args.warmup + args.repetitions):
                times = await controller.run_stages(stages)
                if repetition >= args.warmup:
                    samples.append(sum(times))
            result[f"{part}_network_s"] = samples
            wire_after = await controller.ifstats(args.iface)
            # Interface counters per actor over warm-up + repetitions (includes TCP/IP
            # overhead and the small control traffic); divide by `iterations`.
            result.setdefault("wire_bytes", {})[part] = dict(
                iterations=args.warmup + args.repetitions,
                actors={actor: {key: (None if wire_after[actor][key] is None
                                      else wire_after[actor][key] - wire_before[actor][key])
                                for key in ("rx_bytes", "tx_bytes")} for actor in wire_after})
            expected = sum(size for stage in stages for _, _, size in stage) * (args.warmup + args.repetitions)
            received = sum(s["bytes_received"] for s in (await controller.stats()).values())
            if received != expected:
                raise RuntimeError(f"{part}: agents received {received} bytes, ledger expects {expected}")
    finally:
        await controller.close()
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
