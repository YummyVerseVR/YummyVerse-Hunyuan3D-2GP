"""One-shot, image-only worker for Hunyuan3D.

The deliberately small amount of validation here happens before importing the
controller.  This makes malformed jobs cheap to reject on machines without the
model dependencies (or a GPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any

from PIL import Image


class _NullLogger:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _read_image(path: Path) -> Image.Image:
    """Decode the complete image, accepting only PNG and JPEG."""
    try:
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("source image must be PNG or JPEG")
            image.verify()
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"invalid source image: {path}") from exc


def _validate_glb(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError("generated file is not a complete GLB")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2:
        raise ValueError("generated file is not a GLB v2")
    if declared_length != len(data):
        raise ValueError("GLB length does not match its header")

    offset = 12
    json_chunk: bytes | None = None
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(data) or chunk_length % 4:
            raise ValueError("invalid GLB chunk length")
        chunk = data[offset:end]
        if json_chunk is None:
            if chunk_type != 0x4E4F534A:  # JSON
                raise ValueError("GLB must begin with a JSON chunk")
            json_chunk = chunk
        offset = end
    if offset != len(data) or json_chunk is None:
        raise ValueError("GLB has no JSON chunk")
    try:
        asset = json.loads(json_chunk.rstrip(b" \t\r\n\0"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("GLB JSON chunk is invalid") from exc
    if not isinstance(asset, dict) or not isinstance(asset.get("asset"), dict):
        raise ValueError("GLB JSON has no asset object")
    if asset["asset"].get("version") != "2.0":
        raise ValueError("GLB asset version is not 2.0")
    return data


def _atomic_publish(source: Path, destination: Path) -> bytes:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as target, source.open("rb") as incoming:
            while chunk := incoming.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        _handoff_to_spool_owner(Path(temporary), destination.parent)
        os.replace(temporary, destination)
        return digest.digest()
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _handoff_to_spool_owner(path: Path, spool: Path) -> None:
    """Make root-created bind-mount outputs readable by the host worker."""

    if os.geteuid() != 0:
        return
    owner = spool.stat(follow_symlinks=False)
    os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one GLB from one image")
    parser.add_argument("image_arg", nargs="?")
    parser.add_argument("output_arg", nargs="?")
    parser.add_argument("config_arg", nargs="?")
    parser.add_argument("manifest_arg", nargs="?")
    parser.add_argument("--source-image", "--image", dest="image")
    parser.add_argument("--output-glb", "--output", dest="output")
    parser.add_argument("--config-json", "--config", dest="config")
    parser.add_argument("--manifest", dest="manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    image_path = Path(args.image or args.image_arg) if (args.image or args.image_arg) else None
    output_path = Path(args.output or args.output_arg) if (args.output or args.output_arg) else None
    config_path = Path(args.config or args.config_arg) if (args.config or args.config_arg) else None
    manifest_path = Path(args.manifest or args.manifest_arg) if (args.manifest or args.manifest_arg) else None
    if None in (image_path, output_path, config_path, manifest_path):
        _parser().error("image, output GLB, config JSON, and manifest are required")
    assert image_path and output_path and config_path and manifest_path

    image = _read_image(image_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Keep this import below all input validation: importing controller loads
    # torch, CUDA extensions, and the model pipeline dependencies.
    from controller import Hunyuan3DController

    controller = Hunyuan3DController(config, _NullLogger())
    returned, _ = controller.generate(image=image)
    if not returned:
        raise RuntimeError("controller returned no GLB path")
    returned_path = Path(returned)
    _validate_glb(returned_path)
    digest = _atomic_publish(returned_path, output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_type": "GLB",
        "media_type": "model/gltf-binary",
        "path": output_path.name,
        "sha256": digest.hex(),
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _handoff_to_spool_owner(temporary, manifest_path.parent)
    os.replace(temporary, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
