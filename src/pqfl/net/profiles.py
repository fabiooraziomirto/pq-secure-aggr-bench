"""Network emulation profiles and the `tc` commands that apply them (never executed here)."""
import math

import yaml


def load_profiles(path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        profiles = yaml.safe_load(handle)["profiles"]
    for name, profile in profiles.items():
        rtt, rate, loss = profile["rtt_ms"], profile["rate_mbit"], profile["loss_pct"]
        if not isinstance(rtt, (int, float)) or rtt < 0 or not math.isfinite(rtt):
            raise ValueError(f"{name}: rtt_ms must be a nonnegative number")
        if rate is not None and (not isinstance(rate, (int, float)) or rate <= 0):
            raise ValueError(f"{name}: rate_mbit must be positive or null")
        if not isinstance(loss, (int, float)) or not 0 <= loss < 100:
            raise ValueError(f"{name}: loss_pct must lie in [0, 100)")
    return profiles


def tc_commands(profile: dict, interface: str) -> list[str]:
    """Commands to run as root inside one container. Empty list means no emulation."""
    if not interface.isidentifier() and not all(c.isalnum() or c in "-._" for c in interface):
        raise ValueError("Invalid interface name")
    delay, rate, loss = profile["rtt_ms"] / 2, profile["rate_mbit"], profile["loss_pct"]
    if delay == 0 and rate is None and loss == 0:
        return []
    commands = [f"tc qdisc del dev {interface} root 2>/dev/null || true"]
    netem = f"tc qdisc add dev {interface} root handle 1: netem"
    if delay:
        netem += f" delay {delay:g}ms"
    if loss:
        netem += f" loss {loss:g}%"
    commands.append(netem)
    if rate is not None:
        commands.append(f"tc qdisc add dev {interface} parent 1:1 handle 10: tbf "
                        f"rate {rate:g}mbit burst 256kb latency 400ms")
    return commands
