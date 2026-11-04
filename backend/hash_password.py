"""Generate the login hash / JWT secret for .env (values are printed only to your terminal).

  python -m backend.hash_password                 # prompts for a password, prints SIGHTLINE_DEMO_PASSWORD_HASH=...
  python -m backend.hash_password --admin         # same for SIGHTLINE_ADMIN_PASSWORD_HASH
  python -m backend.hash_password --generate      # makes a random demo password, prints it once + its hash
  python -m backend.hash_password --secret        # prints SIGHTLINE_JWT_SECRET=<random 64 chars>
"""
from __future__ import annotations

import argparse
import getpass
import secrets
import sys

from .security import hash_password


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.hash_password")
    ap.add_argument("--admin", action="store_true", help="print the admin variable name")
    ap.add_argument("--generate", action="store_true", help="generate a random password")
    ap.add_argument("--secret", action="store_true", help="print a random SIGHTLINE_JWT_SECRET")
    a = ap.parse_args(argv)
    if a.secret:
        print(f"SIGHTLINE_JWT_SECRET={secrets.token_urlsafe(48)}")
        return 0
    var = "SIGHTLINE_ADMIN_PASSWORD_HASH" if a.admin else "SIGHTLINE_DEMO_PASSWORD_HASH"
    if a.generate:
        pw = "-".join(secrets.token_urlsafe(4) for _ in range(3))
        print(f"password (give this to judges, it is not stored anywhere): {pw}", file=sys.stderr)
    else:
        pw = getpass.getpass("password: ")
        if pw != getpass.getpass("again: "):
            print("passwords differ", file=sys.stderr)
            return 1
        if len(pw) < 8:
            print("use at least 8 characters", file=sys.stderr)
            return 1
    print(f"{var}={hash_password(pw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
