import numpy as np
import torch

from motor_thermal import data as D
from motor_thermal.models import CausalCNN, GRUNet, RidgeBaseline, count_params


def test_cnn_shapes_and_budget():
    m = CausalCNN()
    out = m(torch.randn(3, 200, 8))
    assert out.shape == (3, 200, 4)
    assert count_params(m) < 100_000
    assert m.receptive_field == 1023


def test_gru_shapes_and_budget():
    m = GRUNet()
    assert m(torch.randn(2, 50, 8)).shape == (2, 50, 4)
    assert count_params(m) < 100_000


def _perturb_check(model, T=300, t0=200):
    torch.manual_seed(0)
    # make the residual branches non-trivial (they are zero-initialised)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2) if p.dim() > 1 else None
    x = torch.randn(1, T, 8)
    x2 = x.clone()
    x2[:, t0:] += 10.0  # perturb the future only
    model.eval()
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert torch.allclose(a[:, :t0], b[:, :t0], atol=1e-5)
    assert not torch.allclose(a[:, t0:], b[:, t0:], atol=1e-3)


def test_cnn_causal():
    _perturb_check(CausalCNN(channels=8, n_blocks=5))


def test_gru_causal():
    _perturb_check(GRUNet(hidden=8, layers=2))


def test_cnn_receptive_field_is_finite():
    m = CausalCNN(channels=8, n_blocks=4).eval()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.3)
    rf = m.receptive_field
    x = torch.randn(1, 100, 8)
    x2 = x.clone()
    x2[:, 0] += 100.0
    with torch.no_grad():
        a, b = m(x), m(x2)
    # outputs further than rf-1 steps away from the perturbation are unchanged
    assert torch.allclose(a[:, rf:], b[:, rf:], atol=1e-5)


def test_ridge_fits_and_is_causal():
    rng = np.random.default_rng(0)
    xs = [rng.normal(size=(400, 8)).astype(np.float32) for _ in range(3)]
    ys = [np.cumsum(x[:, :4], 0) * 0.01 for x in xs]
    m = RidgeBaseline(alpha=1.0, spans=(5, 50)).fit(xs, ys)
    p = m.predict(xs[0])
    assert p.shape == (400, 4)
    x2 = xs[0].copy()
    x2[300:] += 20
    p2 = m.predict(x2)
    np.testing.assert_allclose(p[:300], p2[:300], atol=1e-4)
    assert m.n_params == 4 * (8 + 2) * 3 + 4
    assert D.ridge_features(xs[0], (5, 50)).shape[1] == 30


def test_onnx_export_matches_torch_and_predict_profile(tmp_path):
    import pandas as pd

    from motor_thermal.export import check_match, export_onnx, make_session
    from motor_thermal.predict import predict_profile

    torch.manual_seed(0)
    m = CausalCNN(channels=8, n_blocks=5, kernel=2).eval()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.2)
    path = tmp_path / "m.onnx"
    export_onnx(m, path, T=64)
    sess = make_session(path)
    for shape in [(1, 64, 8), (2, 100, 8)]:
        x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
        assert check_match(m, sess, x) < 1e-4

    sc = D.Scaler(np.zeros(8), np.ones(8), np.zeros(4), np.full(4, 2.0))
    sc.to_json(tmp_path / "s.json")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(150, 8)), columns=D.INPUT_COLS)
    df["profile_id"] = [1] * 80 + [2] * 70
    out = predict_profile(df, path, tmp_path / "s.json")
    assert list(out.columns) == D.TARGET_COLS and len(out) == 150
    # profiles are processed separately: profile 2 alone gives the same predictions
    alone = predict_profile(
        df[df.profile_id == 2].reset_index(drop=True), path, tmp_path / "s.json"
    )
    np.testing.assert_allclose(out.iloc[80:].to_numpy(), alone.to_numpy(), atol=1e-4)
    with torch.no_grad():
        ref = m(torch.from_numpy(df[D.INPUT_COLS].to_numpy(np.float32)[:80])[None])[0].numpy() * 2.0
    np.testing.assert_allclose(out.iloc[:80].to_numpy(), ref, atol=1e-4)
