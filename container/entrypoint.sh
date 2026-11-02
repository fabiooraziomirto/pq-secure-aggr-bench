#!/bin/sh
# Applies the emulated egress profile (needs NET_ADMIN), then runs the command.
# Default command: the network agent for $ACTOR on port 9000.
set -eu
if [ "${NETEM_PROFILE:-lan}" != "lan" ]; then
    python scripts/netem_commands.py --profile "$NETEM_PROFILE" --interface "${NETEM_IFACE:-eth0}" | sh
fi
if [ "$#" -eq 0 ]; then
    set -- python scripts/net_agent.py --actor "$ACTOR" --host 0.0.0.0 --port 9000
fi
exec "$@"
