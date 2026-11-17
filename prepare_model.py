"""Prepare the pinned SAM 2.1 Tiny model without loading any model code."""

import argparse
from hashlib import file_digest
from pathlib import Path
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "sam2.1-hiera-tiny"
REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
BASE = f"https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/{REVISION}"
FILES = {
    "config.json": (5695, "860aff9751b139d83a4ad7df1e5535416fded533e0ead02625edbefcb9953cce"),
    "model.safetensors": (155908064, "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7"),
    "preprocessor_config.json": (683, "6ebf229ee259368ce4a8d4f2fe893a72b053023710853e257253939e601f583d"),
    "processor_config.json": (95, "f8a68e865cfad115c1c2763f3d93eca7b1c622da06da2a9273eb437fb2389b6d"),
}


def verified(path: Path) -> bool:
    size, digest = FILES[path.name]
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest() == digest


def prepare(source: Path | None = None) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sam2-", dir=MODEL_DIR.parent) as work:
        staged = []
        for name in FILES:
            destination = MODEL_DIR / name
            if verified(destination):
                continue
            temporary = Path(work) / name
            if source is None:
                print(f"Downloading {name}", flush=True)
                subprocess.run(["curl.exe", "-fsSL", "--retry", "2", "--connect-timeout", "15",
                                "--max-time", "600", "--output", str(temporary), f"{BASE}/{name}"], check=True)
            else:
                copyfile(source / name, temporary)
            if not verified(temporary):
                raise ValueError(f"Model file failed SHA-256 verification: {name}")
            staged.append((temporary, destination))
        for temporary, destination in staged:
            temporary.replace(destination)
    print(f"SAM 2.1 Tiny is ready in {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dir", type=Path, help="copy already downloaded official model files")
    prepare(parser.parse_args().from_dir)
