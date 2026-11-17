"""Install one pinned MM Grounding DINO snapshot and its public smoke-test images."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory

from grounding import FILES, HEALTH_EXAMPLES, MODEL_DIR, digest


REVISION = "80b3081e63ff7c265a9b404296d848f0f0aa0006"
MODEL_URL = f"https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg/resolve/{REVISION}"
HEALTH_URL = "https://s3.amazonaws.com/open-images-dataset/train"


def prepare(destination, source=None):
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [(Path(name), expected, f"{MODEL_URL}/{name}") for name, expected in FILES.items()]
    files.extend((Path("health") / row["file"], row["sha256"], f"{HEALTH_URL}/{row['file']}")
                 for row in HEALTH_EXAMPLES)
    with TemporaryDirectory(prefix="dino-", dir=destination.parent) as work:
        staging = Path(work)
        pending = []
        for name, expected, url in files:
            target = destination / name
            if target.is_file() and digest(target) == expected:
                continue
            temporary = staging / name
            temporary.parent.mkdir(parents=True, exist_ok=True)
            if source is None:
                print(f"Downloading {name}", flush=True)
                subprocess.run(["curl.exe", "-fsSL", "--retry", "2", "--connect-timeout", "15",
                                "--max-time", "600", "--output", str(temporary), url], check=True)
            else:
                copyfile(source / name, temporary)
            if digest(temporary) != expected:
                raise ValueError(f"DINO file failed SHA-256 verification: {name}")
            pending.append((temporary, target))
        for temporary, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(target)
        manifest = staging / "manifest.json"
        manifest.write_text(json.dumps(HEALTH_EXAMPLES, indent=2) + "\n", encoding="utf-8")
        manifest.replace(destination / "health/manifest.json")
    print(f"MM Grounding DINO is ready in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=MODEL_DIR)
    parser.add_argument("--from-dir", type=Path, help="copy verified model files and health images without downloading")
    args = parser.parse_args()
    prepare(args.destination, args.from_dir)
