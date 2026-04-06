import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import urlopen


def compute_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as out_file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)


def ensure_local_rfdetr_import_path() -> None:
    local_site_packages = Path(__file__).resolve().parent / "rfdetr_venv" / "Lib" / "site-packages"
    if local_site_packages.exists() and str(local_site_packages) not in sys.path:
        sys.path.insert(0, str(local_site_packages))


def collect_model_assets(include_platform: bool = False) -> list[dict[str, Any]]:
    ensure_local_rfdetr_import_path()
    model_weights_module = importlib.import_module("rfdetr.assets.model_weights")
    ModelWeights = getattr(model_weights_module, "ModelWeights")

    assets: list[dict[str, Any]] = []
    for model in ModelWeights:
        assets.append(
            {
                "filename": model.filename,
                "url": model.url,
                "expected_md5": model.md5_hash,
                "source": "ModelWeights",
            }
        )

    if include_platform:
        try:
            platform_module = importlib.import_module("rfdetr.platform.platform_downloads")
            PLATFORM_MODELS = getattr(platform_module, "PLATFORM_MODELS")

            known_filenames = {a["filename"] for a in assets}
            for filename, url in PLATFORM_MODELS.items():
                if filename not in known_filenames:
                    assets.append(
                        {
                            "filename": filename,
                            "url": url,
                            "expected_md5": None,
                            "source": "PLATFORM_MODELS",
                        }
                    )
        except Exception:
            pass

    return assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RF-DETR model weights and record MD5 values."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rfdetr_weights"),
        help="Directory to store downloaded weights",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rfdetr_weights_md5_manifest.json"),
        help="Path to output JSON manifest",
    )
    parser.add_argument(
        "--include-platform-models",
        action="store_true",
        help="Also include legacy platform-only model URLs when available",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if target file already exists",
    )

    args = parser.parse_args()

    assets = collect_model_assets(include_platform=args.include_platform_models)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    for asset in assets:
        filename = asset["filename"]
        url = asset["url"]
        expected_md5 = asset["expected_md5"]
        target_path = args.output_dir / filename

        entry: dict[str, Any] = {
            "filename": filename,
            "url": url,
            "source": asset["source"],
            "expected_md5": expected_md5,
            "path": str(target_path.resolve()),
        }

        try:
            if args.force or not target_path.exists():
                print(f"Downloading: {filename}")
                download_file(url, target_path)
            else:
                print(f"Skipping existing: {filename}")

            actual_md5 = compute_md5(target_path)
            entry["actual_md5"] = actual_md5
            entry["size_bytes"] = target_path.stat().st_size
            entry["status"] = "ok"
            entry["md5_match"] = (
                actual_md5 == expected_md5 if expected_md5 is not None else None
            )
        except (HTTPError, URLError, OSError) as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)

        manifest.append(entry)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
