import numpy as np

from motor_thermal.evaluate import aggregate_seeds, max_abs_error, mean_std, rmse, summarize


def test_rmse_and_max():
    y = np.zeros((4, 2))
    p = np.array([[1, 0], [-1, 0], [1, 0], [-3, 2.0]])
    np.testing.assert_allclose(rmse(y, p), [np.sqrt(3), 1.0])
    np.testing.assert_allclose(max_abs_error(y, p), [3, 2])


def test_summarize_means_over_targets():
    y = np.zeros((3, 4))
    p = np.tile([1.0, 2.0, 3.0, 4.0], (3, 1))
    s = summarize(y, p)
    assert s["rmse_mean"] == 2.5 and s["max_mean"] == 2.5
    assert s["rmse_per_target"] == [1.0, 2.0, 3.0, 4.0]


def test_mean_std_uses_sample_std():
    r = mean_std([1.0, 2.0, 3.0])
    assert r["mean"] == 2.0 and abs(r["std"] - 1.0) < 1e-12
    assert mean_std([5.0])["std"] == 0.0


def test_aggregate_seeds():
    a = {
        "rmse_mean": 1.0,
        "max_mean": 4.0,
        "rmse_per_target": [1, 1, 1, 1],
        "max_per_target": [4, 4, 4, 4],
    }
    b = {
        "rmse_mean": 3.0,
        "max_mean": 6.0,
        "rmse_per_target": [3, 3, 3, 3],
        "max_per_target": [6, 6, 6, 6],
    }
    g = aggregate_seeds([a, b])
    assert g["rmse_mean"]["mean"] == 2.0
    assert g["max_per_target"]["mean"] == [5.0] * 4
    assert abs(g["rmse_mean"]["std"] - np.sqrt(2)) < 1e-12
