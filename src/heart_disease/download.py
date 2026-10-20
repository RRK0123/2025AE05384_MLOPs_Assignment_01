"""Download and validate the original UCI Cleveland file."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen

from .config import ROOT
from .data import clean_data, read_raw
from .schema import SOURCE_PAGE, SOURCE_URL


def download(force: bool = False):
    directory = ROOT / "data" / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "processed.cleveland.data"
    manifest_path = directory / "source.json"
    if path.exists() and not force:
        clean_data(read_raw(path))
        if not manifest_path.exists():
            raise ValueError("Raw file exists without source.json; use --force to refresh.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest["sha256"]:
            raise ValueError("Raw checksum mismatch. Inspect changes before re-downloading.")
        print(f"Using validated cached data: {path}")
        return path
    request = Request(SOURCE_URL, headers={"User-Agent": "MLOps-Assignment-EDA/1.0"})
    with urlopen(request, timeout=45) as response:
        content = response.read()
    temporary = directory / "processed.cleveland.download"
    temporary.write_bytes(content)
    try:
        raw = read_raw(temporary)
        clean_data(raw)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "dataset": "UCI Heart Disease - processed Cleveland subset",
        "source_page": SOURCE_PAGE,
        "download_url": SOURCE_URL,
        "doi": "10.24432/C52P4X",
        "license": "CC BY 4.0",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(raw),
        "source_columns": list(raw.columns),
        "binary_target_rule": "target = 1 if num > 0 else 0",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded and validated {len(raw)} rows: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refresh cached source from UCI.")
    download(parser.parse_args().force)


if __name__ == "__main__":
    main()
