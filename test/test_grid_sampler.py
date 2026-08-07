import numpy as np
import pytest

from ppmat.datasets.grid_sampler import DensityGridSampler


def _field(density, grid_coord=None):
    density = np.asarray(density, dtype=np.float32)
    if grid_coord is None:
        grid_coord = np.arange(density.shape[0] * 3, dtype=np.float32).reshape(-1, 3)
    return {
        "density": density,
        "grid_coord": np.asarray(grid_coord, dtype=np.float32),
    }


def test_sampler_requires_keyword_arguments():
    with pytest.raises(TypeError, match="positional"):
        DensityGridSampler(8)


@pytest.mark.parametrize("n_samples", [0, -1])
def test_sampler_requires_positive_sample_count(n_samples):
    with pytest.raises(ValueError, match="positive"):
        DensityGridSampler(n_samples=n_samples)


def test_sampler_rejects_non_integer_sample_count():
    with pytest.raises(TypeError, match="integer"):
        DensityGridSampler(n_samples=1.5)


@pytest.mark.parametrize(
    "params",
    [{"importance_ratio": 1.5}, {"extreme_ratio": -0.1}],
)
def test_sampler_rejects_out_of_range_ratios(params):
    with pytest.raises(ValueError, match="between 0 and 1"):
        DensityGridSampler(n_samples=4, **params)


def test_sampler_rejects_unsupported_sampling_mode():
    with pytest.raises(ValueError, match="Unsupported sampling_mode"):
        DensityGridSampler(n_samples=4, sampling_mode="nearest")


def test_fixed_random_sampling_requires_seed():
    with pytest.raises(ValueError, match="sampling_seed"):
        DensityGridSampler(
            n_samples=1,
            sampling_mode="random",
            resample_each_epoch=False,
        )


def test_fixed_uniform_sampling_does_not_require_seed():
    sampler = DensityGridSampler(n_samples=1, resample_each_epoch=False)

    data = sampler(_field([1.0, 2.0]), 0)

    np.testing.assert_array_equal(data["density"], [1.0])


def test_sampler_rejects_length_mismatch():
    field = _field([1.0, 2.0], grid_coord=np.zeros([1, 3], dtype=np.float32))

    with pytest.raises(ValueError, match="must match"):
        DensityGridSampler(n_samples=1)(field, 0)


def test_sampler_rejects_empty_field():
    with pytest.raises(ValueError, match="empty density field"):
        DensityGridSampler(n_samples=1)(_field([]), 0)


def test_identity_fixes_the_draw_when_not_resampling_each_epoch():
    params = {
        "n_samples": 8,
        "sampling_mode": "random",
        "sampling_seed": 2026,
        "resample_each_epoch": False,
    }
    density = np.linspace(0.0, 1.0, 20, dtype=np.float32)

    first = DensityGridSampler(**params).indices(density, 3)
    second = DensityGridSampler(**params).indices(density, 3)
    other_index = DensityGridSampler(**params).indices(density, 4)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other_index)


@pytest.mark.parametrize("sampling_mode", ["random", "importance"])
def test_fixed_draw_is_reproducible_for_every_mode(sampling_mode):
    params = {
        "n_samples": 8,
        "sampling_mode": sampling_mode,
        "sampling_seed": 2026,
        "resample_each_epoch": False,
    }
    density = np.linspace(0.0, 1.0, 20, dtype=np.float32)

    first = DensityGridSampler(**params)(_field(density), 1)
    second = DensityGridSampler(**params)(_field(density), 1)

    np.testing.assert_array_equal(first["density"], second["density"])
    np.testing.assert_array_equal(first["grid_coord"], second["grid_coord"])


def test_uniform_sampling_is_deterministic_without_random_offset():
    sampler = DensityGridSampler(n_samples=4)
    density = np.arange(20, dtype=np.float32)

    first = sampler.indices(density, 0)
    second = sampler.indices(density, 7)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, [0, 6, 12, 19])


def test_uniform_random_offset_draws_one_point_per_bin():
    sampler = DensityGridSampler(
        n_samples=8,
        uniform_random_offset=True,
        sampling_seed=4,
        resample_each_epoch=False,
    )
    density = np.arange(20, dtype=np.float32)

    indices = sampler.indices(density, 0)
    bin_edges = np.linspace(0, 20, num=9, dtype=int)

    assert np.unique(indices).size == indices.size
    assert np.all((indices >= bin_edges[:-1]) & (indices < bin_edges[1:]))


def test_oversampling_repeats_points():
    sampler = DensityGridSampler(n_samples=5)

    data = sampler(_field([1.0, 2.0]), 0)

    assert data["density"].shape == (5,)
    assert np.unique(data["density"]).size < 5


def test_indices_are_sorted():
    sampler = DensityGridSampler(
        n_samples=6,
        sampling_mode="random",
        sampling_seed=11,
        resample_each_epoch=False,
    )

    indices = sampler.indices(np.arange(50, dtype=np.float32), 2)

    np.testing.assert_array_equal(indices, np.sort(indices))


def test_resampling_each_epoch_follows_the_numpy_global_rng():
    """The DataLoader reseeds NumPy per worker per epoch, so draws must vary."""

    sampler = DensityGridSampler(
        n_samples=4,
        sampling_mode="random",
        sampling_seed=42,
    )
    density = np.arange(40, dtype=np.float32)

    np.random.seed(0)
    first_pass = [sampler.indices(density, i).tolist() for i in range(4)]
    np.random.seed(1)
    second_pass = [sampler.indices(density, i).tolist() for i in range(4)]
    np.random.seed(0)
    replayed = [sampler.indices(density, i).tolist() for i in range(4)]

    assert first_pass != second_pass
    assert first_pass == replayed
    assert len({tuple(draw) for draw in first_pass}) == len(first_pass)


def test_extreme_ratio_above_importance_ratio_stays_within_high_quota():
    density = np.concatenate(
        [
            np.zeros(50, dtype=np.float32),
            np.ones(50, dtype=np.float32),
        ]
    )
    sampler = DensityGridSampler(
        n_samples=10,
        sampling_mode="importance",
        importance_threshold=1e-5,
        extreme_threshold=0.5,
        importance_ratio=0.2,
        extreme_ratio=0.9,
        sampling_seed=3,
        resample_each_epoch=False,
    )

    assert sampler.indices(density, 0).shape == (10,)


def test_importance_sampling_ranks_multi_channel_density_by_max_channel():
    density = np.zeros([6, 2], dtype=np.float32)
    density[4, 1] = 10.0
    density[5, 0] = 20.0
    sampler = DensityGridSampler(
        n_samples=2,
        sampling_mode="importance",
        importance_threshold=1.0,
        importance_ratio=1.0,
        sampling_seed=1,
        resample_each_epoch=False,
    )

    data = sampler(
        _field(density, grid_coord=np.zeros([6, 3], dtype=np.float32)),
        0,
    )

    np.testing.assert_array_equal(
        data["density"],
        np.asarray([[0.0, 10.0], [20.0, 0.0]], dtype=np.float32),
    )


def test_importance_sampling_prefers_points_above_threshold():
    density = np.concatenate(
        [
            np.zeros(90, dtype=np.float32),
            np.full(10, 5.0, dtype=np.float32),
        ]
    )
    sampler = DensityGridSampler(
        n_samples=10,
        sampling_mode="importance",
        importance_threshold=1.0,
        importance_ratio=1.0,
        sampling_seed=5,
        resample_each_epoch=False,
    )

    data = sampler(_field(density), 0)

    np.testing.assert_array_equal(data["density"], np.full(10, 5.0, dtype=np.float32))


def test_fixed_sampling_identity_is_stable_and_seed_controllable():
    density = np.arange(256, dtype=np.float32)
    identity = ("test", 7, "sample.cube")
    params = {
        "n_samples": 24,
        "sampling_mode": "random",
        "sampling_seed": 2026,
        "resample_each_epoch": False,
    }

    sampler = DensityGridSampler(**params)
    replay = DensityGridSampler(**params)
    first = sampler.indices(density, identity)
    second = replay.indices(density, identity)

    np.testing.assert_array_equal(first, second)
    for changed_identity in (
        ("validation", 7, "sample.cube"),
        ("test", 8, "sample.cube"),
        ("test", 7, "other.cube"),
    ):
        changed = sampler.indices(density, changed_identity)
        assert not np.array_equal(first, changed)

    changed_seed = DensityGridSampler(
        **{**params, "sampling_seed": params["sampling_seed"] + 1}
    ).indices(density, identity)
    assert not np.array_equal(first, changed_seed)


def test_importance_sampling_fills_high_quota_when_extreme_points_dominate():
    density = np.concatenate(
        [
            np.zeros(80, dtype=np.float32),
            np.ones(20, dtype=np.float32),
        ]
    )
    sampler = DensityGridSampler(
        n_samples=20,
        sampling_mode="importance",
        importance_threshold=0.5,
        importance_ratio=0.8,
        extreme_threshold=0.5,
        extreme_ratio=0.05,
        sampling_seed=7,
        resample_each_epoch=False,
    )

    indices = sampler.indices(density, ("train", 0, "sample"))

    assert np.count_nonzero(density[indices] >= 0.5) >= 16
