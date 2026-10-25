import json

import numpy as np
import pandas as pd
import pytest

from motor_thermal import data as D


def synthetic_df(n_profiles=6, length=300, seed=0, shuffle_ids=True):
    rng = np.random.default_rng(seed)
    pids = [58, 65, 72, 2, 3, 4][:n_profiles]
    if shuffle_ids:
        pids = pids[::-1]
    frames = []
    for pid in pids:
        t = pd.DataFrame(rng.normal(size=(length, 8)) + pid, columns=D.INPUT_COLS)
        for c in D.TARGET_COLS:
            t[c] = rng.normal(size=length) - pid
        t[D.PROFILE_COL] = pid
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def test_split_p1():
    ids = D.split_profile_ids(np.array([2, 3, 58, 65, 72, 5]))
    assert ids["val"] == [58]
    assert ids["test"] == [65, 72]
    assert ids["train"] == [2, 3, 5]
    assert not set(ids["train"]) & (set(ids["val"]) | set(ids["test"]))


def test_profiles_keep_time_order_and_do_not_mix():
    df = synthetic_df()
    prof = D.profiles_from_df(df)
    for pid, (x, y) in prof.items():
        assert x.shape == (300, 8) and y.shape == (300, 4)
        # every row of that profile has inputs centred on its id (we added pid to the inputs)
        assert abs(x.mean() - pid) < 0.3
        sub = df[df[D.PROFILE_COL] == pid]
        np.testing.assert_array_equal(x, sub[D.INPUT_COLS].to_numpy(np.float32))


def test_windows_never_cross_profiles():
    prof = D.profiles_from_df(synthetic_df())
    x, y, pids, starts = D.build_windows(prof, length=64, stride=32)
    assert len(x) == len(pids) > 0
    for xi, yi, pid, s in zip(x, y, pids, starts, strict=True):
        px, py = prof[int(pid)]
        np.testing.assert_array_equal(xi, px[s : s + 64])
        np.testing.assert_array_equal(yi, py[s : s + 64])
        assert s + 64 <= len(px)


def test_inputs_contain_no_targets():
    assert set(D.INPUT_COLS).isdisjoint(D.TARGET_COLS)
    prof = D.profiles_from_df(synthetic_df())
    x, y = prof[58]
    assert x.shape[1] == 8
    # no input column equals a target column
    for j in range(8):
        for k in range(4):
            assert not np.allclose(x[:, j], y[:, k])
    # ridge features depend on inputs only: changing targets cannot change them
    f1 = D.ridge_features(x)
    f2 = D.ridge_features(x.copy())
    np.testing.assert_array_equal(f1, f2)
    assert f1.shape[1] == (8 + 2) * (1 + len(D.DEFAULT_SPANS))


def test_scaler_fit_on_train_only(tmp_path):
    prof = D.profiles_from_df(synthetic_df())
    ids = D.split_profile_ids(list(prof))
    train = {i: prof[i] for i in ids["train"]}
    sc = D.Scaler.fit(train)
    xtr = np.concatenate([p[0] for p in train.values()])
    np.testing.assert_allclose(sc.input_mean, xtr.mean(0), rtol=1e-5)
    np.testing.assert_allclose(sc.input_std, xtr.std(0), rtol=1e-5)
    # held-out profiles do not influence it
    all_sc = D.Scaler.fit(prof)
    assert not np.allclose(all_sc.input_mean, sc.input_mean)
    # round trip
    p = tmp_path / "scaler.json"
    sc.to_json(p)
    sc2 = D.Scaler.from_json(p)
    np.testing.assert_allclose(sc2.target_std, sc.target_std)
    assert json.loads(p.read_text())["input_names"] == D.INPUT_COLS
    y = prof[58][1]
    np.testing.assert_allclose(sc.inverse_y(sc.y(y)), y, rtol=1e-4, atol=1e-4)


def test_ewma_is_causal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 8))
    base = D.ridge_features(x)
    x2 = x.copy()
    x2[120:] += 50.0  # perturb the future
    pert = D.ridge_features(x2)
    np.testing.assert_allclose(base[:120], pert[:120], rtol=1e-6, atol=1e-6)
    assert not np.allclose(base[120:], pert[120:])


def test_ewma_values():
    x = np.array([[1.0], [3.0], [5.0]])
    out = D.ewma(x, span=3)  # alpha = 0.5
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 3.5])


def test_random_window_split_is_leaky_by_construction():
    prof = D.profiles_from_df(synthetic_df())
    _, _, pids, _ = D.build_windows(prof, 64, 16)
    tr, va, te = D.random_window_split(pids, seed=0)
    assert len(set(tr) | set(va) | set(te)) == len(pids)
    # the same profile appears on both sides of the split
    assert set(pids[tr]) & set(pids[te])


def test_load_raw_npz_and_csv(tmp_path):
    df = synthetic_df(n_profiles=3, length=20)
    cols = D.INPUT_COLS + D.TARGET_COLS
    np.savez(
        tmp_path / "emt_f32.npz",
        X=df[cols].to_numpy(np.float32),
        columns=np.array(cols),
        profile_id=df[D.PROFILE_COL].to_numpy(np.int16),
    )
    a = D.load_raw(tmp_path)
    df.to_csv(tmp_path / "measures_v2.csv", index=False)
    b = D.load_raw(tmp_path / "measures_v2.csv")
    np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), rtol=1e-5)
    assert (a[D.PROFILE_COL].to_numpy() == df[D.PROFILE_COL].to_numpy()).all()
    with pytest.raises(FileNotFoundError):
        D.load_raw(tmp_path / "nope.csv")
