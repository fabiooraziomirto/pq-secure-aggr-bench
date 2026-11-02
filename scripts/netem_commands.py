#!/usr/bin/env python3
"""Print the tc commands for a profile; pipe to `sh` inside a container with NET_ADMIN.

Prints only; applies nothing. Example: netem_commands.py --profile wan --interface eth0 | sh
"""
import argparse

from pqfl.net.profiles import load_profiles, tc_commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="configs/network_profiles.yaml")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--interface", required=True)
    args = parser.parse_args()
    profiles = load_profiles(args.profiles)
    if args.profile not in profiles:
        parser.error(f"unknown profile; choose from {sorted(profiles)}")
    print("\n".join(tc_commands(profiles[args.profile], args.interface)))


if __name__ == "__main__":
    main()
