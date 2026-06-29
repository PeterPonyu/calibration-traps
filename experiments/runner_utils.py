"""Small helpers shared by experiment grid runners.

The runners are intentionally simple sequential loops.  `--num-shards/--shard-id`
lets two or more machines split the same deterministic cell list without adding a
database, scheduler, or shared filesystem lock.  Default args preserve the
original single-runner behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Sequence, TypeVar

T = TypeVar("T")


def add_shard_args(ap: ArgumentParser) -> None:
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the planned cell list into N deterministic shards (default: 1).",
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Run shard K where 0 <= K < --num-shards (default: 0).",
    )


def validate_shard_args(args: Namespace) -> None:
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit("--shard-id must satisfy 0 <= shard-id < num-shards")


def shard_cells(cells: Sequence[T], num_shards: int, shard_id: int) -> list[T]:
    """Return the deterministic modulo shard of a cell list."""
    if num_shards == 1:
        return list(cells)
    return [cell for i, cell in enumerate(cells) if i % num_shards == shard_id]


def shard_suffix(num_shards: int, shard_id: int, total: int, selected: int) -> str:
    if num_shards == 1:
        return ""
    return f" | shard {shard_id}/{num_shards}: {selected}/{total} selected"
