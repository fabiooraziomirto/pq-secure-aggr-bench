#!/usr/bin/env python3
"""Run one actor's network agent (one per container). Opt-in; never run by tests."""
import argparse
import asyncio

from pqfl.net.replay import Agent


async def serve(actor, host, port):
    agent = Agent(actor)
    bound = await agent.start(host, port)
    print(f"agent {actor} listening on {host}:{bound}", flush=True)
    await asyncio.Event().wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--host", required=True, help="Bind address, e.g. 0.0.0.0 in a container")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(serve(args.actor, args.host, args.port))


if __name__ == "__main__":
    main()
