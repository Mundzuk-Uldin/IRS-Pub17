"""Issue a JWT for the pub17 service.

Signs with PUB17_JWT_SECRET. The token is printed so it can go in a curl
header; the secret never is.

Usage:  python3 scripts/mint_token.py --role preparer --sub alice [--ttl 3600]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pub17 import auth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=sorted(auth.ROLES))
    ap.add_argument("--sub", required=True, help="who the token is for; logged with each request")
    ap.add_argument("--ttl", type=int, default=3600, help="seconds until expiry")
    args = ap.parse_args()
    print(auth.mint(args.sub, args.role, args.ttl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
