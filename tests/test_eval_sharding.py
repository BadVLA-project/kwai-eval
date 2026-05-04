import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_sharding_module():
    spec = importlib.util.spec_from_file_location(
        'sharding_for_test',
        ROOT / 'vlmeval' / 'utils' / 'sharding.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shuffle_sharding_is_deterministic_and_preserves_all_items(monkeypatch):
    module = load_sharding_module()
    monkeypatch.delenv('VLMEVAL_SHARD_STRATEGY', raising=False)
    monkeypatch.delenv('VLMEVAL_SHARD_SEED', raising=False)
    items = list(range(24))

    shards = [module.shard_items(items, rank=rank, world_size=4) for rank in range(4)]
    shards_again = [module.shard_items(items, rank=rank, world_size=4) for rank in range(4)]

    assert shards == shards_again
    assert shards != [items[rank::4] for rank in range(4)]
    assert sorted(item for shard in shards for item in shard) == items


def test_round_robin_strategy_keeps_existing_order(monkeypatch):
    module = load_sharding_module()
    monkeypatch.setenv('VLMEVAL_SHARD_STRATEGY', 'round_robin')

    assert module.shard_items(list(range(10)), rank=1, world_size=3) == [1, 4, 7]
