"""Single time source so tests can freeze and advance time."""
from datetime import datetime, timezone


def now() -> datetime:
    return datetime.now(timezone.utc)
