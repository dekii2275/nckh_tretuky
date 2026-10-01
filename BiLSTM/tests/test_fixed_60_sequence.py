import numpy as np

from src.data.dataset import uniform_temporal_resample


def test_uniform_resample_always_returns_60_frames():
    sequence = np.arange(5 * 25 * 2, dtype=np.float32).reshape(5, 25, 2)
    result = uniform_temporal_resample(sequence, 60)
    assert result.shape == (60, 25, 2)
    assert np.allclose(result[0], sequence[0])
    assert np.allclose(result[-1], sequence[-1])
    assert np.isfinite(result).all()


def test_uniform_downsample_is_deterministic_and_preserves_endpoints():
    sequence = np.arange(120 * 25 * 2, dtype=np.float32).reshape(120, 25, 2)
    first = uniform_temporal_resample(sequence, 60)
    second = uniform_temporal_resample(sequence, 60)
    assert first.shape == (60, 25, 2)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], sequence[0])
    assert np.array_equal(first[-1], sequence[-1])
