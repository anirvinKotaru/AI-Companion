from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_girlfriend.avatar.face import (
    AvatarAssets,
    _load_cached_frames,
    _save_cached_frames,
)


def _fake_assets() -> AvatarAssets:
    return AvatarAssets(
        background=np.full((4, 4, 3), 10, dtype=np.uint8),
        mouth_frames={
            "idle": np.full((4, 4, 4), 20, dtype=np.uint8),
            "A": np.full((4, 4, 4), 30, dtype=np.uint8),
        },
        eye_overlays={
            "left": np.full((4, 4, 4), 40, dtype=np.uint8),
            "right": np.full((4, 4, 4), 50, dtype=np.uint8),
        },
    )


def _write_image(path: Path) -> None:
    path.write_bytes(b"not a real image, only mtime/size matter to the cache")


def test_round_trips_through_the_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "avatar.webp"
    _write_image(image_path)
    assets = _fake_assets()

    _save_cached_frames(image_path, assets)
    loaded = _load_cached_frames(image_path)

    assert loaded is not None
    assert np.array_equal(loaded.background, assets.background)
    assert loaded.mouth_frames.keys() == assets.mouth_frames.keys()
    for name in assets.mouth_frames:
        assert np.array_equal(loaded.mouth_frames[name], assets.mouth_frames[name])
    assert loaded.eye_overlays.keys() == assets.eye_overlays.keys()
    for eye_id in assets.eye_overlays:
        assert np.array_equal(loaded.eye_overlays[eye_id], assets.eye_overlays[eye_id])


def test_missing_cache_file_is_a_cache_miss(tmp_path: Path) -> None:
    image_path = tmp_path / "avatar.webp"
    _write_image(image_path)
    assert _load_cached_frames(image_path) is None


def test_image_changing_after_the_cache_was_written_invalidates_it(tmp_path: Path) -> None:
    image_path = tmp_path / "avatar.webp"
    _write_image(image_path)
    _save_cached_frames(image_path, _fake_assets())

    # A later mtime and different size, as if the user swapped in a new photo.
    image_path.write_bytes(b"a completely different (and longer) fake image")

    assert _load_cached_frames(image_path) is None


def test_stale_cache_version_is_a_cache_miss(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "avatar.webp"
    _write_image(image_path)
    _save_cached_frames(image_path, _fake_assets())

    import ai_girlfriend.avatar.face as face

    monkeypatch.setattr(face, "_CACHE_VERSION", face._CACHE_VERSION + 1)
    assert _load_cached_frames(image_path) is None
