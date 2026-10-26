import numpy as np
import pytest
import torch

from munl import SISAConfig, SISAEnsemble, make_synthetic, prepare

SMALL = dict(num_shards=3, num_slices=4, epochs_per_slice=2, batch_size=32, hidden=(32, 16), seed=7)


@pytest.fixture(scope="module")
def data():
    return prepare(make_synthetic(600, seed=1), seed=1)


def fit(data, **overrides):
    ens = SISAEnsemble(SISAConfig(**{**SMALL, **overrides}), data.n_features)
    return ens.fit(data.train.X, data.train.y, data.train.ids)


def test_partition_covers_every_row_exactly_once(data):
    ens = fit(data)
    held = np.concatenate([p.ids for shard in ens.parts for p in shard])
    assert sorted(held) == sorted(data.train.ids)
    assert len(ens) == len(data.train.ids)


def test_checkpoints_are_independent_snapshots(data):
    ens = fit(data)
    first, last = ens.checkpoints[0][0]["model"], ens.checkpoints[0][-1]["model"]
    assert any(not torch.equal(first[k], last[k]) for k in first)


def test_training_is_deterministic(data):
    a, b = fit(data), fit(data)
    np.testing.assert_array_equal(a.predict(data.test.X), b.predict(data.test.X))


def test_unlearning_only_retrains_the_affected_shard(data):
    ens = fit(data)
    rid = int(data.train.ids[0])
    shard, slice_idx = ens.location[rid]
    untouched = {s: [p.clone() for p in ens.models[s].parameters()] for s in range(SMALL["num_shards"]) if s != shard}

    report = ens.unlearn([rid])

    assert rid not in ens
    assert report.retrained_from == {shard: slice_idx}
    assert report.slices_retrained == SMALL["num_slices"] - slice_idx
    for s, params in untouched.items():
        assert all(torch.equal(a, b) for a, b in zip(params, ens.models[s].parameters()))


def test_unlearning_is_exact(data):
    """Train-then-unlearn must equal never having trained on the record."""
    rid = int(data.train.ids[10])
    ens = fit(data)
    ens.unlearn([rid])

    ref = SISAEnsemble(SISAConfig(**SMALL), data.n_features)
    ref.fit(data.train.X, data.train.y, data.train.ids, exclude=[rid])

    np.testing.assert_array_equal(ens.predict(data.test.X), ref.predict(data.test.X))


def test_batched_unlearning_and_unknown_ids(data):
    ens = fit(data)
    ids = [int(i) for i in data.train.ids[:5]]
    report = ens.unlearn(ids + [ids[0], -1])
    assert report.removed == 5
    assert report.not_found == [-1]
    assert len(ens) == len(data.train.ids) - 5
    assert all(i not in ens for i in ids)


def test_save_load_roundtrip(data, tmp_path):
    ens = fit(data, aggregation="median")
    path = tmp_path / "sisa.pt"
    ens.save(path)
    loaded = SISAEnsemble.load(path)
    np.testing.assert_array_equal(ens.predict(data.test.X), loaded.predict(data.test.X))

    rid = int(data.train.ids[3])
    ens.unlearn([rid])
    loaded.unlearn([rid])
    np.testing.assert_array_equal(ens.predict(data.test.X), loaded.predict(data.test.X))


def test_record_number_is_not_a_feature(data):
    # 6 numeric columns + one-hot categoricals; the id column must not leak in.
    names = data.preprocessor.get_feature_names_out()
    assert not any("Record Number" in n for n in names)


def test_model_beats_predicting_the_mean():
    data = prepare(make_synthetic(3000, seed=2), seed=2)
    ens = SISAEnsemble(SISAConfig(num_shards=2, num_slices=2, epochs_per_slice=10, hidden=(64, 32)), data.n_features)
    ens.fit(data.train.X, data.train.y, data.train.ids)
    mse = float(np.mean((ens.predict(data.test.X) - data.test.y) ** 2))
    assert mse < 0.6 * float(np.var(data.test.y))
