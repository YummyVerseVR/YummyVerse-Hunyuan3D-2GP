import hashlib
import importlib.util
import json
import struct
import sys
import types

from PIL import Image


def load_worker():
    spec = importlib.util.spec_from_file_location("worker_cli", "src/worker_cli.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_glb():
    payload = json.dumps({"asset": {"version": "2.0"}}, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    chunk = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


def test_image_worker_publishes_glb_and_manifest(tmp_path, monkeypatch):
    worker = load_worker()
    image_path = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(image_path)
    returned = tmp_path / "generated.glb"
    returned.write_bytes(tiny_glb())
    calls = {}

    class FakeController:
        def __init__(self, config, logger):
            calls["config"] = config

        def generate(self, **kwargs):
            calls["image"] = kwargs["image"]
            return str(returned), None

    monkeypatch.setitem(sys.modules, "controller", types.SimpleNamespace(Hunyuan3DController=FakeController))
    config = tmp_path / "config.json"
    config.write_text('{"hunyuan3d": {"device": "cpu"}}')
    output = tmp_path / "out" / "model.glb"
    manifest = tmp_path / "out" / "manifest.json"

    assert worker.main([str(image_path), str(output), str(config), str(manifest)]) == 0
    assert output.read_bytes() == returned.read_bytes()
    document = json.loads(manifest.read_text())
    assert document == {
        "artifact_type": "GLB",
        "media_type": "model/gltf-binary",
        "path": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    assert calls["image"].size == (1, 1)


def test_root_container_hands_outputs_to_spool_owner(tmp_path, monkeypatch):
    worker = load_worker()
    image_path = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(image_path)
    returned = tmp_path / "generated.glb"
    returned.write_bytes(tiny_glb())

    class FakeController:
        def __init__(self, _config, _logger):
            pass

        def generate(self, **_kwargs):
            return str(returned), None

    monkeypatch.setitem(sys.modules, "controller", types.SimpleNamespace(Hunyuan3DController=FakeController))
    monkeypatch.setattr(worker.os, "geteuid", lambda: 0)
    handoffs = []
    monkeypatch.setattr(
        worker.os,
        "chown",
        lambda path, uid, gid, **kwargs: handoffs.append(
            (path, uid, gid, kwargs)
        ),
    )
    config = tmp_path / "config.json"
    config.write_text('{"hunyuan3d": {"device": "cpu"}}')
    output = tmp_path / "out" / "model.glb"
    manifest = tmp_path / "out" / "manifest.json"

    assert worker.main([str(image_path), str(output), str(config), str(manifest)]) == 0
    owner = output.parent.stat()
    assert len(handoffs) == 2
    assert {(uid, gid) for _, uid, gid, _ in handoffs} == {
        (owner.st_uid, owner.st_gid)
    }
    assert all(options == {"follow_symlinks": False} for *_, options in handoffs)


def test_invalid_image_is_rejected_before_controller_import(tmp_path, monkeypatch):
    worker = load_worker()
    image_path = tmp_path / "bad.png"
    image_path.write_bytes(b"not an image")
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.delitem(sys.modules, "controller", raising=False)

    try:
        worker.main([str(image_path), str(tmp_path / "out.glb"), str(config), str(tmp_path / "manifest.json")])
    except ValueError as exc:
        assert "invalid source image" in str(exc)
    else:
        raise AssertionError("invalid image was accepted")
    assert "controller" not in sys.modules


def test_parser_exposes_no_caption_or_t23d_options():
    worker = load_worker()
    options = worker._parser().format_help()
    assert "caption" not in options
    assert "t23d" not in options.lower()
