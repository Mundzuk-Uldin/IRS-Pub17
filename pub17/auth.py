"""JWT auth: three roles, each scoped to the document collections it may query.

Tokens are HS256, signed with PUB17_JWT_SECRET, and carry `sub` (who) and
`role`. There is no login endpoint and no user table -- scripts/mint_token.py
issues tokens -- because the point here is scoping and audit, not identity
management.
"""
import os
import time

import jwt

ALGORITHM = "HS256"

# Collections are groups of source files. Retrieval for a request is filtered
# to the files its role can see, so a scoped user can't pull a chunk from a
# collection outside the scope, even as a citation.
COLLECTIONS = {
    "publications": ["p17.pdf", "p501.pdf", "p502.pdf"],
    "forms": ["i1040gi.pdf", "f1040s1.pdf", "f1040s1a.pdf", "f1040s2.pdf", "f1040s3.pdf"],
}

ROLES = {
    # Plain-language guidance only.
    "viewer": {"collections": ["publications"], "admin": False},
    # Guidance plus form instructions and schedules.
    "preparer": {"collections": ["publications", "forms"], "admin": False},
    # Everything, plus the request log and spend endpoints.
    "admin": {"collections": ["publications", "forms"], "admin": True},
}


# RFC 7518 3.2: an HS256 key should be at least as long as the hash output.
MIN_SECRET_BYTES = 32


def _secret():
    secret = os.environ.get("PUB17_JWT_SECRET")
    if not secret:
        raise RuntimeError("PUB17_JWT_SECRET is not set")
    if len(secret.encode()) < MIN_SECRET_BYTES:
        raise RuntimeError(f"PUB17_JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes")
    return secret


def mint(subject, role, ttl_seconds=3600):
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLES)}")
    now = int(time.time())
    claims = {"sub": subject, "role": role, "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(claims, _secret(), algorithm=ALGORITHM)


def verify(token):
    """Return the claims of a valid token. Raises jwt.InvalidTokenError otherwise."""
    claims = jwt.decode(token, _secret(), algorithms=[ALGORITHM], options={"require": ["sub", "role", "exp"]})
    if claims["role"] not in ROLES:
        raise jwt.InvalidTokenError(f"unknown role {claims['role']!r}")
    return claims


def collections_for(role):
    return ROLES[role]["collections"]


def sources_for(role):
    return [f for c in collections_for(role) for f in COLLECTIONS[c]]


def is_admin(role):
    return ROLES[role]["admin"]
