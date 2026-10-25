"""Tests for scripts/download_data.py with kagglehub mocked."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


@pytest.fixture()
def dl():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_kagglehub(monkeypatch, fn):
    mod = types.ModuleType("kagglehub")
    mod.dataset_download = fn
    monkeypatch.setitem(sys.modules, "kagglehub", mod)


def test_download_copies_file(dl, tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache" / "v1"
    cache.mkdir(parents=True)
    (cache / "measures_v2.csv").write_text("u_q,pm\n1,2\n")
    calls = []

    def fake(name):
        calls.append(name)
        return str(cache)

    _fake_kagglehub(monkeypatch, fake)
    dest = tmp_path / "data"
    assert dl.main(["--dest", str(dest)]) == 0
    assert calls == ["wkirgsn/electric-motor-temperature"]
    assert (dest / "measures_v2.csv").read_text() == "u_q,pm\n1,2\n"
    out = capsys.readouterr().out
    assert "measures_v2.csv" in out
    assert "doi" in out


def test_link_option(dl, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "measures_v2.csv").write_text("a\n")
    _fake_kagglehub(monkeypatch, lambda name: str(cache))
    dest = tmp_path / "data"
    assert dl.main(["--dest", str(dest), "--link"]) == 0
    assert (dest / "measures_v2.csv").is_symlink()


def test_missing_kagglehub(dl, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "kagglehub", None)  # makes the import fail
    assert dl.main(["--dest", str(tmp_path)]) == 1
    assert "kagglehub is not installed" in capsys.readouterr().err


def test_credentials_error(dl, tmp_path, monkeypatch, capsys):
    def boom(name):
        raise RuntimeError("401 Unauthorized")

    _fake_kagglehub(monkeypatch, boom)
    assert dl.main(["--dest", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "credentials" in err and "401" in err


def test_file_missing_in_dataset(dl, tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    cache.mkdir()
    _fake_kagglehub(monkeypatch, lambda name: str(cache))
    assert dl.main(["--dest", str(tmp_path / "d")]) == 1
    assert "not found" in capsys.readouterr().err
