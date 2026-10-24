#!/usr/bin/env python3
"""Print public environment metadata without triggering native auto-installation."""
import json
from pqfl.logging.environment import environment_metadata

if __name__ == '__main__':
    print(json.dumps(environment_metadata(), indent=2))
