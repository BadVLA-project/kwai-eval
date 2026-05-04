import os
import random
from typing import Iterable, TypeVar


T = TypeVar('T')


def _shard_strategy() -> str:
    strategy = os.environ.get('VLMEVAL_SHARD_STRATEGY', 'shuffle').strip().lower()
    aliases = {
        'rr': 'round_robin',
        'round-robin': 'round_robin',
        'ordered': 'round_robin',
        'none': 'round_robin',
    }
    return aliases.get(strategy, strategy)


def _shard_seed() -> int:
    return int(os.environ.get('VLMEVAL_SHARD_SEED', '0'))


def shard_items(items: Iterable[T], rank: int, world_size: int) -> list[T]:
    """Return this rank's deterministic shard of a shared item order."""
    if world_size <= 0:
        raise ValueError('world_size must be positive')
    if not 0 <= rank < world_size:
        raise ValueError(f'rank must be in [0, {world_size}), got {rank}')

    ordered = list(items)
    if world_size == 1:
        return ordered

    strategy = _shard_strategy()
    if strategy == 'round_robin':
        return ordered[rank::world_size]
    if strategy != 'shuffle':
        raise ValueError(
            f'Unsupported VLMEVAL_SHARD_STRATEGY={strategy!r}; '
            'expected shuffle or round_robin'
        )

    shuffled = ordered[:]
    random.Random(_shard_seed()).shuffle(shuffled)
    return shuffled[rank::world_size]
