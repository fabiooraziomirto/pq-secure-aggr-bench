"""Replay a payload-free transfer ledger over real TCP sockets.

The aggregators are in-process simulators; their ledgers list (sender, receiver,
size) with a causal stage. Replaying a ledger between actor agents measures the
network cost of the exact byte pattern, stage by stage. Payloads are zeros: the
network cost depends only on sizes. Per-transfer time runs from the first byte
written to the receiver's 1-byte acknowledgement, so it includes the return
half-RTT of that ack. Stage time is the slowest transfer of the stage; round time
is the sum of stages. Compute time is NOT included.
"""
import asyncio
import json
import time
from collections import defaultdict

CHUNK = 1 << 16
_ZEROS = bytes(CHUNK)


def ledger_stages(stats) -> list[list[tuple[str, str, int]]]:
    """Group a RoundStats ledger into ordered stages of concurrent transfers."""
    if len(stats.stages) != len(stats.transfers):
        raise ValueError("Ledger has no stage information")
    grouped = defaultdict(list)
    for transfer, stage in zip(stats.transfers, stats.stages):
        grouped[stage].append((transfer.sender, transfer.receiver, transfer.size))
    return [grouped[stage] for stage in sorted(grouped)]


class Agent:
    """One actor: accepts data frames and executes `send` commands."""

    def __init__(self, actor: str):
        self.actor = actor
        self.bytes_received = 0
        self.bytes_sent = 0
        self._connections = {}
        self._locks = defaultdict(asyncio.Lock)
        self._server = None

    async def start(self, host: str, port: int) -> int:
        self._server = await asyncio.start_server(self._handle, host, port)
        return self._server.sockets[0].getsockname()[1]

    async def close(self):
        for _, writer in self._connections.values():
            writer.close()
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer):
        try:
            while True:
                kind = await reader.read(1)
                if not kind:
                    return
                if kind == b"D":
                    size = int.from_bytes(await reader.readexactly(8), "big")
                    remaining = size
                    while remaining:
                        remaining -= len(await reader.readexactly(min(CHUNK, remaining)))
                    self.bytes_received += size
                    writer.write(b"A")
                    await writer.drain()
                elif kind == b"C":
                    command = json.loads(await reader.readline())
                    writer.write((json.dumps(await self._command(command)) + "\n").encode())
                    await writer.drain()
                else:
                    raise ValueError("Unknown frame type")
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return
        finally:
            writer.close()

    async def _command(self, command):
        if command["op"] == "stats":
            return {"actor": self.actor, "bytes_sent": self.bytes_sent,
                    "bytes_received": self.bytes_received}
        if command["op"] == "sysctl":
            names = ("tcp_congestion_control", "tcp_slow_start_after_idle")
            values = {}
            for name in names:
                try:
                    with open(f"/proc/sys/net/ipv4/{name}", encoding="ascii") as handle:
                        values[name] = handle.read().strip()
                except OSError:
                    values[name] = None
            return values
        if command["op"] == "ifstats":
            wanted = command.get("iface", "eth0")
            try:
                with open("/proc/net/dev", encoding="ascii") as handle:
                    for line in handle.read().splitlines()[2:]:
                        name, counters = line.split(":", 1)
                        if name.strip() == wanted:
                            fields = counters.split()
                            return {"rx_bytes": int(fields[0]), "tx_bytes": int(fields[8])}
            except OSError:
                pass
            return {"rx_bytes": None, "tx_bytes": None}
        if command["op"] == "reset":
            self.bytes_sent = self.bytes_received = 0
            return {"ok": True}
        if command["op"] == "connect":
            await self._connection(command["host"], command["port"])
            return {"ok": True}
        if command["op"] == "send":
            return {"elapsed": await self._send(command["host"], command["port"], command["size"])}
        raise ValueError("Unknown command")

    async def _connection(self, host, port):
        key = (host, port)
        if key not in self._connections:
            self._connections[key] = await asyncio.open_connection(host, port)
        return self._connections[key]

    async def _send(self, host, port, size) -> float:
        reader, writer = await self._connection(host, port)
        async with self._locks[(host, port)]:  # one frame at a time per connection
            start = time.perf_counter()
            writer.write(b"D" + size.to_bytes(8, "big"))
            remaining = size
            while remaining:
                count = min(CHUNK, remaining)
                writer.write(_ZEROS[:count])
                remaining -= count
                await writer.drain()
            await reader.readexactly(1)
            elapsed = time.perf_counter() - start
        self.bytes_sent += size
        return elapsed


class Controller:
    """Drives agents; addresses maps actor -> (host, port)."""

    def __init__(self, addresses: dict[str, tuple[str, int]]):
        self.addresses = addresses

    async def _ask(self, actor, command):
        # One short-lived control connection per command, so concurrent commands to
        # the same agent (a broadcast) really run concurrently. Timing is taken
        # inside the agent and does not include control-connection setup.
        reader, writer = await asyncio.open_connection(*self.addresses[actor])
        try:
            writer.write(b"C" + (json.dumps(command) + "\n").encode())
            await writer.drain()
            return json.loads(await reader.readline())
        finally:
            writer.close()

    async def close(self):
        return None

    async def prepare(self, stages):
        """Open every needed connection and reset counters (excluded from timings)."""
        for actor in self.addresses:
            await self._ask(actor, {"op": "reset"})
        edges = {(s, r) for stage in stages for s, r, _ in stage}
        await asyncio.gather(*(self._ask(s, {"op": "connect", "host": self.addresses[r][0],
                                             "port": self.addresses[r][1]}) for s, r in edges))

    async def run_stages(self, stages) -> list[float]:
        times = []
        for stage in stages:
            results = await asyncio.gather(*(self._ask(sender, {
                "op": "send", "host": self.addresses[receiver][0],
                "port": self.addresses[receiver][1], "size": size})
                for sender, receiver, size in stage))
            times.append(max(result["elapsed"] for result in results))
        return times

    async def sysctl(self, actor):
        return await self._ask(actor, {"op": "sysctl"})

    async def ifstats(self, iface="eth0"):
        return {actor: await self._ask(actor, {"op": "ifstats", "iface": iface}) for actor in self.addresses}

    async def stats(self):
        return {actor: await self._ask(actor, {"op": "stats"}) for actor in self.addresses}
