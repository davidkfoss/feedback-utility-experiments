"""Build, verify and fetch a flat, immutable paper-results ZIP."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile
from scripts.paper.records import ROOT, DATA, manifest, load_records

RELEASE = ROOT / "paper/release.json"


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pack(source_root, destination):
    spec = manifest()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staged = Path(tmp) / destination.name
        with zipfile.ZipFile(
            staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as z:
            for r in sorted(spec["runs"], key=lambda r: r["file"]):
                source = Path(source_root) / r["source"]
                raw = source.read_bytes()
                if hashlib.sha256(raw).hexdigest() != r["sha256"]:
                    raise ValueError(f"Changed source: {source}")
                info = zipfile.ZipInfo(r["file"], date_time=(2026, 9, 15, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, raw, compresslevel=9)
        os.replace(staged, destination)
    digest = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n"
    )
    return digest


def unpack(archive, destination, expected_digest):
    archive = Path(archive)
    destination = Path(destination)
    if sha256(archive) != expected_digest:
        raise ValueError("Archive checksum mismatch")
    spec = manifest()
    expected = {r["file"]: r for r in spec["runs"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(expected) or set(names) != set(expected):
                raise ValueError(
                    "ZIP must contain exactly the 205 flat manifest filenames"
                )
            for info in z.infolist():
                r = expected[info.filename]
                if info.file_size != r["size_bytes"]:
                    raise ValueError(f"Unexpected uncompressed size: {info.filename}")
                raw = z.read(info)
                if hashlib.sha256(raw).hexdigest() != r["sha256"]:
                    raise ValueError(f"Invalid result: {info.filename}")
                (stage / info.filename).write_bytes(raw)
        load_records(stage)
        if destination.exists():
            load_records(destination)
            return  # Existing verified archive is already complete; never overwrite raw results.
        os.replace(stage, destination)


def fetch(destination=DATA, archive=None, url=None):
    release = json.loads(RELEASE.read_text())
    if Path(destination).exists():
        load_records(destination)
        return
    if archive:
        unpack(archive, destination, release["sha256"])
        return
    local = ROOT / "dist" / release["asset"]
    if local.exists():
        unpack(local, destination, release["sha256"])
        return
    url = url or release.get("url")
    if not url:
        raise ValueError(
            "Release has not been published yet. Supply --archive PATH or --url URL."
        )
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / release["asset"]
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        unpack(archive, destination, release["sha256"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["fetch", "verify", "pack"])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--archive", type=Path)
    p.add_argument("--url")
    p.add_argument("--source-root", type=Path)
    args = p.parse_args()
    if args.action == "verify":
        load_records(args.data)
        print("Verified 205 paper results")
    elif args.action == "fetch":
        fetch(args.data, args.archive, args.url)
        print("Paper results ready")
    else:
        if not args.source_root or not args.archive:
            p.error("pack requires --source-root and --archive")
        print(pack(args.source_root, args.archive))


if __name__ == "__main__":
    main()
