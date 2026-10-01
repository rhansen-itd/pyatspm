"""Tests for the atspm video-sync CLI handler.

Target: src/atspm/cli.py — handle_video_sync, _add_video_sync_parser.
"""

from pathlib import Path
import pytest

from atspm import cli
from atspm.analysis.video_sync import SyncResult
from atspm.data.video import ShapeConfig


@pytest.fixture
def cli_env(tmp_path):
    target_name = "201_Main"
    target_dir = tmp_path / target_name
    target_dir.mkdir()

    db_path = target_dir / "201.db"
    db_path.touch()

    video_path = target_dir / "clip.mp4"
    video_path.touch()

    shape_path = target_dir / "cam_shapes.csv"
    shapes = [
        {
            "type": "lamp",
            "points": [(50, 50)],
            "color": (0, 255, 0),
            "input": None,
            "phase": "2",
            "name": "Phase 2 green",
            "indication": "green",
        }
    ]
    ShapeConfig(shapes=shapes, video_width=100, video_height=100).save(shape_path)

    return {
        "target_name": target_name,
        "target_dir": target_dir,
        "db_path": db_path,
        "video_path": video_path,
        "shape_path": shape_path,
    }


def test_video_sync_accepted_output(monkeypatch, capsys, cli_env):
    monkeypatch.setattr(cli, "_resolve_target_name", lambda t, tid: cli_env["target_name"])
    monkeypatch.setattr(cli, "_get_target_dir", lambda name: cli_env["target_dir"])
    monkeypatch.setattr(cli, "_load_metadata", lambda tdir: {"timezone": "America/Denver"})
    monkeypatch.setattr(cli, "_resolve_db_path", lambda tdir, meta: cli_env["db_path"])
    monkeypatch.setattr(cli, "_video_shape_path", lambda tdir, cam: cli_env["shape_path"])
    monkeypatch.setattr(cli, "_resolve_video_path", lambda tdir, vid: cli_env["video_path"])

    accepted_result = SyncResult(
        accepted=True,
        reason=None,
        start_epoch=1780000003.3,
        mid_start_epoch=1780000003.3,
        slip_s_per_10min=-0.15,
        score=0.95,
        runner_up_score=0.2,
        agreement=0.98,
        n_frames_compared=6000,
        n_edges_used=12,
        gap_clamped=False,
    )
    monkeypatch.setattr("atspm.video.sync.sync_video", lambda *a, **kw: accepted_result)

    parser = cli._build_parser()
    args = parser.parse_args([
        "video-sync",
        "--target", cli_env["target_name"],
        "--camera", "cam",
        "--video", "clip.mp4",
        "--start-guess", "2026-10-01T12:25:00",
    ])
    args.func(args)

    out = capsys.readouterr().out
    assert "Synchronized video start" in out
    assert "Corrected start:" in out
    assert "Delta from guess:" in out
    assert "Mid-clip start:" in out
    assert "Slip:" in out
    assert "Score:" in out
    assert "Agreement:" in out
    assert "atspm video-overlay --target 201_Main --camera cam --video clip.mp4 --start" in out


def test_video_sync_refused_exit_and_fallback(monkeypatch, capsys, cli_env):
    monkeypatch.setattr(cli, "_resolve_target_name", lambda t, tid: cli_env["target_name"])
    monkeypatch.setattr(cli, "_get_target_dir", lambda name: cli_env["target_dir"])
    monkeypatch.setattr(cli, "_load_metadata", lambda tdir: {"timezone": "America/Denver"})
    monkeypatch.setattr(cli, "_resolve_db_path", lambda tdir, meta: cli_env["db_path"])
    monkeypatch.setattr(cli, "_video_shape_path", lambda tdir, cam: cli_env["shape_path"])
    monkeypatch.setattr(cli, "_resolve_video_path", lambda tdir, vid: cli_env["video_path"])

    refused_result = SyncResult(
        accepted=False,
        reason="Score 0.450 below minimum 0.600.",
        start_epoch=1780000003.3,
        mid_start_epoch=1780000003.3,
        slip_s_per_10min=0.0,
        score=0.45,
        runner_up_score=0.2,
        agreement=0.6,
        n_frames_compared=3000,
        n_edges_used=4,
        gap_clamped=False,
    )
    monkeypatch.setattr("atspm.video.sync.sync_video", lambda *a, **kw: refused_result)

    parser = cli._build_parser()
    args = parser.parse_args([
        "video-sync",
        "--target", cli_env["target_name"],
        "--camera", "cam",
        "--video", "clip.mp4",
        "--start-guess", "2026-10-01T12:25:00",
    ])

    with pytest.raises(SystemExit) as exc_info:
        args.func(args)

    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    assert "Video synchronization refused: Score 0.450 below minimum 0.600." in out
    assert "Manual fallback alignment:" in out
    assert "atspm video-locate-phase-change --target 201_Main --camera cam --video clip.mp4 --phase 2 --start" in out


def test_video_sync_missing_lamp_shapes_dies(monkeypatch, capsys, cli_env):
    # Save a config with no lamp shapes
    ShapeConfig(shapes=[], video_width=100, video_height=100).save(cli_env["shape_path"])

    monkeypatch.setattr(cli, "_resolve_target_name", lambda t, tid: cli_env["target_name"])
    monkeypatch.setattr(cli, "_get_target_dir", lambda name: cli_env["target_dir"])
    monkeypatch.setattr(cli, "_load_metadata", lambda tdir: {"timezone": "America/Denver"})
    monkeypatch.setattr(cli, "_resolve_db_path", lambda tdir, meta: cli_env["db_path"])
    monkeypatch.setattr(cli, "_video_shape_path", lambda tdir, cam: cli_env["shape_path"])
    monkeypatch.setattr(cli, "_resolve_video_path", lambda tdir, vid: cli_env["video_path"])

    parser = cli._build_parser()
    args = parser.parse_args([
        "video-sync",
        "--target", cli_env["target_name"],
        "--camera", "cam",
        "--video", "clip.mp4",
        "--start-guess", "2026-10-01T12:25:00",
    ])

    with pytest.raises(SystemExit):
        args.func(args)

    err = capsys.readouterr().err
    assert "lamp" in err
